#include "Model.h"

#include <algorithm>
#include <random>
#include <fstream>
#include <iterator>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;
using weighted_label_ = loss_multiclass_log_per_pixel_weighted_::weighted_label;

// ----------------------------------------------------------------------------------------
const bool do_color_reduction = false;
struct training_sample_lr {
    matrix<gray_pixel> input_image;
    matrix<uint16_t> output_image;
    matrix<weighted_label_> weighted_output_image;
};
struct training_sample_hr {
    matrix<gray_pixel> input_image;
    std::array<matrix<float>, 2> output_image;
};

// ----------------------------------------------------------------------------------------
struct current_state {
    chrono::time_point<chrono::system_clock> first_run_time;
    chrono::time_point<chrono::system_clock> last_run_time;
};

// ----------------------------------------------------------------------------------------
template <typename AbImageType>
matrix<rgb_pixel> concact_channels(const matrix<gray_pixel>& gray_image, const AbImageType& ab_image) {
    matrix<lab_pixel> lab_image(gray_image.nr(), gray_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            lab_image(r, c).l = gray_image(r, c);
            if constexpr (std::is_same<AbImageType, matrix<uint16_t>>::value) {
                lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) >> 4, 4));
                lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) & 0xF, 4));
            } else if constexpr (std::is_same<AbImageType, std::array<matrix<float>, 2>>::value) {
                lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[0](r, c)), 4));
                lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[1](r, c)), 4));
            }
        }
    }
    matrix<rgb_pixel> output;
    assign_image(output, lab_image);
    return output;
}

// Helper function to resize grayscale or color image
template <typename pixel_type>
void scale_image(long nr, long nc, matrix<pixel_type>& dst) {
    matrix<pixel_type> resized(nr, nc);
    resize_image(dst, resized, interpolate_bilinear());
    dlib::assign_image(dst, resized);
}
template <typename pixel_type>
void resize_inplace(matrix<pixel_type>& inout, long size) {
    if (inout.nr() != size || inout.nc() != size) {
        matrix<pixel_type> mem_img;
        mem_img.set_size(size, size);
        resize_image(inout, mem_img);
        inout = mem_img;
    }
}
template <typename pixel_type>
void resize_max(matrix<pixel_type>& in, size_t max_image_dims) {
    size_t width = in.nc(), height = in.nr();
    if (width > max_image_dims || height > max_image_dims) {
        const double resize_factor = std::min(max_image_dims / (double)width, max_image_dims / (double)height);
        const size_t new_width = static_cast<size_t>(width * resize_factor);
        const size_t new_height = static_cast<size_t>(height * resize_factor);
        matrix<pixel_type> size_img(new_height, new_width);
        resize_image(in, size_img);
        in = size_img;
    }
}

// ----------------------------------------------------------------------------------------
matrix<rgb_pixel> lr_generate_image(lr_generator_type& net, const matrix<gray_pixel>& src) {
    matrix<gray_pixel> gray_image = src;
    resize_inplace(gray_image, std_image_size);
    matrix<uint16_t> output = net(gray_image);
    matrix<rgb_pixel> rgb_image = concact_channels(gray_image, output);
    scale_image(src.nr(), src.nc(), rgb_image);

    matrix<lab_pixel> lab_image;
    assign_image(lab_image, rgb_image);
    for (long r = 0; r < lab_image.nr(); ++r) for (long c = 0; c < lab_image.nc(); ++c) lab_image(r, c).l = src(r, c);
    assign_image(rgb_image, lab_image);
    return rgb_image;
}
matrix<rgb_pixel> hr_generate_image(hr_generator_type& net, const matrix<gray_pixel>& src) {
    matrix<gray_pixel> gray_image = src;
    resize_inplace(gray_image, std_image_size);
    std::array<matrix<float>, 2> output = net(gray_image);
    matrix<rgb_pixel> rgb_image = concact_channels(gray_image, output);
    scale_image(src.nr(), src.nc(), rgb_image);
    
    matrix<lab_pixel> lab_image;
    assign_image(lab_image, rgb_image);
    for (long r = 0; r < lab_image.nr(); ++r) for (long c = 0; c < lab_image.nc(); ++c) lab_image(r, c).l = src(r, c);
    assign_image(rgb_image, lab_image);
    return rgb_image;
}

// ----------------------------------------------------------------------------------------
rectangle make_random_cropping_rect(const matrix<rgb_pixel>& img, dlib::rand& rnd) {
    // figure out what rectangle we want to crop from the image
    double mins = 0.9, maxs = 1.0;
    auto scale = mins + rnd.get_random_double() * (maxs - mins);
    auto size = scale * std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()), rnd.get_random_32bit_number() % (img.nr() - rect.height()));
    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------
void randomly_crop_image(
    const matrix<rgb_pixel>& input_image, training_sample_lr& crop, dlib::rand& rnd) {
    const auto rect = make_random_cropping_rect(input_image, rnd);
    const chip_details chip_details(rect, chip_dims(std_image_size, std_image_size));

    matrix<rgb_pixel> src_image;
    extract_image_chip(input_image, chip_details, src_image, interpolate_bilinear());
    if (do_color_reduction) reduce_colors(src_image);
    if (rnd.get_random_double() > 0.5) src_image = fliplr(src_image);

    rgb_image_to_grayscale_image(src_image, crop.input_image);    
    matrix<lab_pixel> lab_image;
    assign_image(lab_image, src_image);
    crop.weighted_output_image.set_size(lab_image.nr(), lab_image.nc());
    crop.output_image.set_size(lab_image.nr(), lab_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            uint16_t quantized_a = quantize_n_bits(lab_image(r, c).a, 4);
            uint16_t quantized_b = quantize_n_bits(lab_image(r, c).b, 4);
            uint16_t label = (quantized_a << 4) | quantized_b;
            float weight = calc_weight(quantized_a, quantized_b, 4, lab_image(r, c).a, lab_image(r, c).b, lab_image(r, c).l / 255.0f);
            crop.output_image(r, c) = label;
            crop.weighted_output_image(r, c) = weighted_label(label, weight);
        }
    }
}
void randomly_crop_image(
    const matrix<rgb_pixel>& input_image, training_sample_hr& crop, dlib::rand& rnd) {
    const auto rect = make_random_cropping_rect(input_image, rnd);
    const chip_details chip_details(rect, chip_dims(std_image_size, std_image_size));

    matrix<rgb_pixel> src_image;
    extract_image_chip(input_image, chip_details, src_image, interpolate_bilinear());
    if (do_color_reduction) reduce_colors(src_image);
    if (rnd.get_random_double() > 0.5) src_image = fliplr(src_image);

    rgb_image_to_grayscale_image(src_image, crop.input_image);
    matrix<lab_pixel> lab_image;
    dlib::assign_image(lab_image, src_image);
    crop.output_image[0].set_size(lab_image.nr(), lab_image.nc());
    crop.output_image[1].set_size(lab_image.nr(), lab_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            crop.output_image[0](r, c) = quantize_n_bits(lab_image(r, c).a, 4);
            crop.output_image[1](r, c) = quantize_n_bits(lab_image(r, c).b, 4);
        }
    }
}

// ----------------------------------------------------------------------------------------
bool is_grayscale(const matrix<rgb_pixel>& image) {
    matrix<lab_pixel> lab_image;
    dlib::assign_image(lab_image, image);
    running_stats<double> a_stats, b_stats;
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            a_stats.add(lab_image(r, c).a);
            b_stats.add(lab_image(r, c).b);
        }
    }
    // If both standard deviations are below a threshold, consider it grayscale
    const double threshold = 3.15;
    return a_stats.stddev() < threshold && b_stats.stddev() < threshold;
}
bool is_two_small(const matrix<rgb_pixel>& image) {
    const size_t min_image_size = 200;
    return (image.nc() < min_image_size || image.nr() < min_image_size);
}

// ----------------------------------------------------------------------------------------
std::atomic<bool> g_interrupted = false;
BOOL WINAPI CtrlHandler(DWORD ctrlType) {
    if (ctrlType == CTRL_C_EVENT) {
        g_interrupted = true;
        return TRUE;
    }
    return FALSE;
}

// ----------------------------------------------------------------------------------------
void normalize_images(const std::string& rootDir) {
    const std::vector<file> training_images = dlib::get_files_in_directory_tree(rootDir, dlib::match_endings(".jpeg .jpg .png"));
    long total_files = training_images.size();
    atomic<long> processed_files = 0;

    for (const auto& file : training_images) {
        try {
            matrix<rgb_pixel> input_image;
            load_image(input_image, file.full_name());
            if (is_grayscale(input_image) || is_two_small(input_image)) {
                std::cout << "Too small or grayscale image, deleting: " << file.full_name() << endl;
                fs::remove(file.full_name()); // Remove small images
            }
            size_t prev_nc = input_image.nc(), prev_nr = input_image.nr();
            resize_max(input_image, 1024);
            if (input_image.nc() != prev_nc && input_image.nr() != prev_nr) {
                save_jpeg(input_image, file.full_name(), 90);
                std::cout << "Normalized: " << file.full_name() << endl; // Resize two large images
            }            
        } catch (...) {
            cerr << "Error processing file: " << file.full_name() << ", deleting" << endl;
            fs::remove(file.full_name()); // Remove bad images
        }
        if ((processed_files++ % 100) == 0) {
            long progress_percentage = static_cast<long>((static_cast<float>(processed_files) / total_files * 100));
            std::cout << "Normalization progress: [" << string(long(progress_percentage / 2), '=') << "] " << progress_percentage << "%" << "\r";
            std::cout.flush();
        }
        if (g_interrupted) break;
    }
    std::cout << endl;
}

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cout << "Usage: Colorify --[norm|train-lr|train-hr|test-lr|test-hr] <directory>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string option = argv[1];
    std::srand(std::time(nullptr));
    dlib::rand rnd(std::rand());
    size_t iteration = 0;    
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    if (option == "--norm") {
        normalize_images(argv[2]);
    } else if (option == "--train-lr") {
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(argv[2], dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (training_images.size() == 0) {
            std::cout << "Didn't find images for the training dataset." << endl;
            return EXIT_FAILURE;
        }
        current_state state;
        state.first_run_time = chrono::system_clock::now();
        state.last_run_time = state.first_run_time;

        // Instantiate both generator        
        const size_t minibatch_size = 24;        
        dlib::rand rnd(time(nullptr));
        set_dnn_prefer_fastest_algorithms();
        lr_generator_type generator;

        // Resume training from last sync file
        if (file_exists("lowres_colorify.dnn")) deserialize("lowres_colorify.dnn") >> generator;

        const double learning_rate = 1e-1;
        const double min_learning_rate = 1e-6;
        const double weight_decay = 1e-4;
        const double momentum = 0.9;
        const long patience = 20000;
        const long update_display = 30;
        const long max_minutes_elapsed = 5;

        // Initialize the trainer
        dnn_trainer<lr_generator_type> trainer(generator, sgd(weight_decay, momentum));
        trainer.set_learning_rate(learning_rate);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_mini_batch_size(minibatch_size);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.be_verbose();
        set_all_bn_running_stats_window_sizes(generator, 1000);

        // Output training parameters
        std::cout << "The network has " << generator.num_layers << " layers in it." << std::endl;
        std::cout << generator << std::endl;
        std::cout << std::endl << trainer << std::endl;
        // Total images in the dataset
        std::cout << "images in dataset: " << training_images.size() << endl;

        // Use some threads to preload images
        dlib::pipe<training_sample_lr> data(minibatch_size);
        auto f = [&data, &training_images](time_t seed) {
            dlib::rand rnd(time(nullptr) + seed);
            matrix<rgb_pixel> input_image;
            training_sample_lr temp;
            while (data.is_enabled()) {
                const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
                try { 
                    load_image(input_image, image_info.full_name());
                    randomly_crop_image(input_image, temp, rnd);
                    data.enqueue(temp);
                } catch (...) {
                    cerr << "Error during image loading: " << image_info.full_name() << endl;
                }                
            }
        };
        std::thread data_loader1([f]() { f(1); });
        std::thread data_loader2([f]() { f(2); });
        std::cout << "Waiting for the initial pipe loading... ";
        while (data.size() < minibatch_size) std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "done" << std::endl;

        std::vector<matrix<uint16_t>> labels;
        std::vector<matrix<weighted_label_>> weighted_labels;
        std::vector<matrix<gray_pixel>> samples;
        training_sample_lr sample;
        dlib::image_window win;
        while (!g_interrupted) {
            // Train
            labels.clear();
            weighted_labels.clear();
            samples.clear();
            while (samples.size() < minibatch_size) {
                data.dequeue(sample);
                labels.push_back(sample.output_image);
                weighted_labels.push_back(sample.weighted_output_image);
                samples.push_back(sample.input_image);
            }            
            if (++iteration % update_display == 0) { // We should see that the generated images start looking like samples
                std::vector<matrix<rgb_pixel>> disp_imgs;
                matrix<rgb_pixel> src_img, rgb_gen, image_to_save;
                trainer.get_net(dlib::force_flush_to_disk::no);
                auto gen_samples = generator(samples);
                size_t pos_i = 0, max_iter = __min(gen_samples.size(), 4);
                for (auto& image : gen_samples) {
                    src_img = concact_channels(samples[pos_i], labels[pos_i]);
                    rgb_gen = concact_channels(samples[pos_i], image);
                    disp_imgs.push_back(join_rows(src_img, rgb_gen));
                    if (++pos_i >= max_iter) break;
                }
                image_to_save = tile_images(disp_imgs);
                save_jpeg(image_to_save, "lowres_model_training.jpg", 90);
                win.set_image(image_to_save);
                win.set_title("COLORIFY - Supervised-learning process, step#: " + to_string(iteration) + " - " + to_string(max_iter) + " samples");
            }
            trainer.train_one_step(samples, weighted_labels);
            // Check if the model has to be saved
            if (iteration % 200 == 0) {
                chrono::time_point<chrono::system_clock> current_time = chrono::system_clock::now();
                double minutes_elapsed = chrono::duration_cast<chrono::minutes>(current_time - state.last_run_time).count();                
                if (minutes_elapsed >= max_minutes_elapsed) {
                    trainer.get_net(dlib::force_flush_to_disk::no);
                    generator.clean();
                    serialize("lowres_colorify.dnn") << generator;
                    cout << "checkpoint#:\tModel <lowres_colorify.dnn> saved on disk" << endl;
                    state.last_run_time = current_time;
                }
            }
        }
        data.disable();
        data_loader1.join();
        data_loader2.join();

        // Once the training has finished, we don't need the discriminator any more. We just keep the generator
        // We also save the checkpoint again to iterate the learning process
        trainer.get_net();
        generator.clean();
        serialize("lowres_colorify.dnn") << generator;
    } else if (option == "--train-hr") {
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(argv[2], dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (training_images.size() == 0) {
            std::cout << "Didn't find images for the training dataset." << endl;
            return EXIT_FAILURE;
        }
        current_state state;
        state.first_run_time = chrono::system_clock::now();
        state.last_run_time = state.first_run_time;

        // Instantiate both generator        
        const size_t minibatch_size = 32;        
        dlib::rand rnd(time(nullptr));
        set_dnn_prefer_fastest_algorithms();
        hr_generator_type generator;

        // Resume training from last sync file
        if (file_exists("highres_colorify.dnn")) deserialize("highres_colorify.dnn") >> generator;

        const double learning_rate = 1e-1;
        const double min_learning_rate = 1e-6;
        const double weight_decay = 1e-4;
        const double momentum = 0.9;
        const long patience = 20000;
        const long update_display = 30;
        const long max_minutes_elapsed = 5;

        // Initialize the trainer
        dnn_trainer<hr_generator_type> trainer(generator, sgd(weight_decay, momentum));
        trainer.set_learning_rate(learning_rate);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_mini_batch_size(minibatch_size);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.be_verbose();
        set_all_bn_running_stats_window_sizes(generator, 1000);

        // Output training parameters
        std::cout << "The network has " << generator.num_layers << " layers in it." << std::endl;
        std::cout << generator << std::endl;
        std::cout << std::endl << trainer << std::endl;
        // Total images in the dataset
        std::cout << "images in dataset: " << training_images.size() << endl;

        // Use some threads to preload images
        dlib::pipe<training_sample_hr> data(minibatch_size);
        auto f = [&data, &training_images](time_t seed) {
            dlib::rand rnd(time(nullptr) + seed);
            matrix<rgb_pixel> input_image;
            training_sample_hr temp;
            while (data.is_enabled()) {
                const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
                try { 
                    load_image(input_image, image_info.full_name());
                    randomly_crop_image(input_image, temp, rnd);
                    data.enqueue(temp);
                } catch (...) {
                    cerr << "Error during image loading: " << image_info.full_name() << endl;
                }                
            }
        };
        std::thread data_loader1([f]() { f(1); });
        std::thread data_loader2([f]() { f(2); });
        std::cout << "Waiting for the initial pipe loading... ";
        while (data.size() < minibatch_size) std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "done" << std::endl;

        std::vector<std::array<matrix<float>, 2>> labels;
        std::vector<matrix<gray_pixel>> samples;
        training_sample_hr sample;
        dlib::image_window win;
        while (!g_interrupted) {
            // Train
            labels.clear();
            samples.clear();
            while (samples.size() < minibatch_size) {
                data.dequeue(sample);
                labels.push_back(sample.output_image);
                samples.push_back(sample.input_image);
            }            
            if (++iteration % update_display == 0) { // We should see that the generated images start looking like samples
                std::vector<matrix<rgb_pixel>> disp_imgs;
                matrix<rgb_pixel> src_img, rgb_gen, image_to_save;
                trainer.get_net(dlib::force_flush_to_disk::no);
                auto gen_samples = generator(samples);
                size_t pos_i = 0, max_iter = __min(gen_samples.size(), 4);
                for (auto& image : gen_samples) {
                    src_img = concact_channels(samples[pos_i], labels[pos_i]);
                    rgb_gen = concact_channels(samples[pos_i], image);
                    disp_imgs.push_back(join_rows(src_img, rgb_gen));
                    if (++pos_i >= max_iter) break;
                }
                image_to_save = tile_images(disp_imgs);
                save_jpeg(image_to_save, "highres_model_training.jpg");
                win.set_image(image_to_save);
                win.set_title("COLORIFY - Supervised-learning process, step#: " + to_string(iteration) + " - " + to_string(max_iter) + " samples");
            }
            trainer.train_one_step(samples, labels);
            // Check if the model has to be saved
            if (iteration % 200 == 0) {
                chrono::time_point<chrono::system_clock> current_time = chrono::system_clock::now();
                double minutes_elapsed = chrono::duration_cast<chrono::minutes>(current_time - state.last_run_time).count();                
                if (minutes_elapsed >= max_minutes_elapsed) {
                    trainer.get_net(dlib::force_flush_to_disk::no);
                    generator.clean();
                    serialize("highres_colorify.dnn") << generator;
                    cout << "checkpoint#:\tModel <highres_colorify.dnn> saved on disk" << endl;
                    state.last_run_time = current_time;
                }
            }
        }
        data.disable();
        data_loader1.join();
        data_loader2.join();

        // Once the training has finished, we don't need the discriminator any more. We just keep the generator
        // We also save the checkpoint again to iterate the learning process
        trainer.get_net();
        generator.clean();
        serialize("highres_colorify.dnn") << generator;
    } else if (option == "--test-lr" || option == "--test-hr") {
        const std::vector<file> images = dlib::get_files_in_directory_tree(argv[2], dlib::match_endings(".jpeg .jpg .png"));
        std::cout << "total images to colorify: " << images.size() << endl;
        if (images.size() == 0) {
            std::cout << "Didn't find images to colorify." << endl;
            return EXIT_FAILURE;
        }

        // Load the mode
        dlib::rand rnd(time(nullptr));
        bool use_lr_model = (option == "--test-lr");
        lr_generator_type lr_generator;
        hr_generator_type hr_generator;
        if (use_lr_model) {
            if (file_exists("lowres_colorify.dnn")) {
                deserialize("lowres_colorify.dnn") >> lr_generator;
            } else {
                std::cout << "Didn't find the model (lowres_colorify.dnn)." << endl;
                return EXIT_FAILURE;
            }
        } else {
            if (file_exists("highres_colorify.dnn")) {
                deserialize("highres_colorify.dnn") >> hr_generator;
            } else {
                std::cout << "Didn't find the model (highres_colorify.dnn)." << endl;
                return EXIT_FAILURE;
            }
        }        

        dlib::image_window win;
        matrix<rgb_pixel> input_image, gen_image, display_gray_image;
        matrix<gray_pixel> gray_image;        
        for (auto& i : images) {            
            try { load_image(input_image, i.full_name()); }
            catch (...) {
                cerr << "Error during image loading: " << i.full_name() << endl;
                continue;
            }
            if (is_grayscale(input_image) || is_two_small(input_image)) continue;
            resize_max(input_image, std_image_size);
            rgb_image_to_grayscale_image(input_image, gray_image);
            assign_image(display_gray_image, gray_image);         

            gen_image = use_lr_model ? lr_generate_image(lr_generator, gray_image) : hr_generate_image(hr_generator, gray_image);
            win.set_title("COLORIFY - Grayscale " + to_string(input_image.nc()) + "x" + to_string(input_image.nr()) + ") | Original | Generated (" + to_string(gen_image.nc()) + "x" + to_string(gen_image.nr()) + ")");
            win.set_image(join_rows(display_gray_image, join_rows(input_image, gen_image)));
            std::cout << i.full_name() << " - Hit enter to process the next image or 'q' to quit";
            char c = std::cin.get();
            if (c == 'q' || c == 'Q') break;
        }
    }
}
catch (std::exception& e)
{
    std::cout << e.what() << endl;
}