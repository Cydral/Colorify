#include "Model.h"

#include <algorithm>
#include <random>
#include <fstream>
#include <iterator>
#include <thread>
#include <filesystem>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;
using weighted_label_ = loss_multiclass_log_per_pixel_weighted_::weighted_label;

using net_backbone = generator_backbone<input<matrix<gray_pixel>>>;
using net_type_lr = loss_multiclass_log_per_pixel_weighted<cont<256, 1, 1, 1, 1, net_backbone>>;
using net_type_hr = loss_mean_squared_per_channel_and_pixel<2, cont<2, 1, 1, 1, 1, net_backbone>>;

// [D] network type
using net_discriminator = loss_binary_log<fc<1, conp<2, 3, 1, 0, dropout<leaky_relu<bn_con<conp<256, 4, 2, 1, dropout<leaky_relu<bn_con<conp<128, 4, 2, 1, leaky_relu<conp<64, 4, 2, 1, input<std::array<matrix<float>, 2>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
const uint8_t lr_res_nb_bits = 4;
const uint8_t hr_res_nb_bits = 5;
const bool do_color_reduction = false;
const uint16_t max_lr_output_value = (1 << (lr_res_nb_bits * 2)) - 1;
const float max_hr_output_value = (1 << hr_res_nb_bits) - 1;
const float avg_hr_output_value = max_hr_output_value / 2.0f;
struct training_sample {
    matrix<gray_pixel> input_image;
    matrix<uint16_t> lr_output_image;
    matrix<weighted_label_> weighted_output_image;
    std::array<matrix<float>, 2> hr_output_image;
};

// ----------------------------------------------------------------------------------------
struct current_state {
    chrono::time_point<chrono::system_clock> first_run_time;
    chrono::time_point<chrono::system_clock> last_run_time;
};

// ----------------------------------------------------------------------------------------
template <typename AbImageType>
matrix<rgb_pixel> concat_channels(const matrix<gray_pixel>& gray_image, const AbImageType& ab_image) {
    matrix<lab_pixel> lab_image(gray_image.nr(), gray_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            lab_image(r, c).l = gray_image(r, c);
            if constexpr (std::is_same<AbImageType, matrix<uint16_t>>::value) {
                if (ab_image(r, c) > max_lr_output_value) {
                    lab_image(r, c).a = lab_image(r, c).b = (1 << lr_res_nb_bits) - 1;
                } else {
                    lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) >> 4, lr_res_nb_bits));
                    lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) & 0xF, lr_res_nb_bits));
                }
            } else if constexpr (std::is_same<AbImageType, std::array<matrix<float>, 2>>::value) {
                if (ab_image[0](r, c) < 0.0f) lab_image(r, c).a = 0;
                else if (ab_image[0](r, c) > max_hr_output_value) lab_image(r, c).a = static_cast<uint8_t>(max_hr_output_value);
                else lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[0](r, c)), hr_res_nb_bits));
                if (ab_image[1](r, c) < 0.0f) lab_image(r, c).b = 0;
                else if (ab_image[1](r, c) > max_hr_output_value) lab_image(r, c).b = static_cast<uint8_t>(max_hr_output_value);                
                else lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[1](r, c)), hr_res_nb_bits));
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
rectangle make_random_cropping_rect(const matrix<rgb_pixel>& img, dlib::rand& rnd, const bool do_augmentation) {    
    if (do_augmentation) {
        // figure out what rectangle we want to crop from the image
        double mins = 0.85, maxs = 1.0;
        auto scale = mins + rnd.get_random_double() * (maxs - mins);
        auto size = scale * std::min(img.nr(), img.nc());
        rectangle rect(size, size);
        // randomly shift the box around
        point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()), rnd.get_random_32bit_number() % (img.nr() - rect.height()));
        return move_rect(rect, offset);
    } else {
        auto size = std::min(img.nr(), img.nc());
        return rectangle(size, size);
    }
}

// ----------------------------------------------------------------------------------------
void randomly_crop_image(const matrix<rgb_pixel>& input_image, training_sample& crop, dlib::rand& rnd, const bool low_res, const bool do_augmentation = false) {
    const auto rect = make_random_cropping_rect(input_image, rnd, do_augmentation);
    const chip_details chip_details(rect, chip_dims(std_image_size, std_image_size));

    matrix<rgb_pixel> src_image;
    extract_image_chip(input_image, chip_details, src_image, interpolate_bilinear());
    if (do_color_reduction) reduce_colors(src_image);
    if (do_augmentation && rnd.get_random_double() > 0.5) src_image = fliplr(src_image);

    rgb_image_to_grayscale_image(src_image, crop.input_image);
    matrix<lab_pixel> lab_image;
    assign_image(lab_image, src_image);
    if (low_res) {        
        crop.weighted_output_image.set_size(lab_image.nr(), lab_image.nc());
        crop.lr_output_image.set_size(lab_image.nr(), lab_image.nc());
        for (long r = 0; r < lab_image.nr(); ++r) {
            for (long c = 0; c < lab_image.nc(); ++c) {
                uint16_t quantized_a = quantize_n_bits(lab_image(r, c).a, lr_res_nb_bits);
                uint16_t quantized_b = quantize_n_bits(lab_image(r, c).b, lr_res_nb_bits);
                uint16_t label = (quantized_a << lr_res_nb_bits) | quantized_b;
                float weight = calc_weight(quantized_a, quantized_b, lr_res_nb_bits, lab_image(r, c).a, lab_image(r, c).b, lab_image(r, c).l / 255.0f);
                crop.lr_output_image(r, c) = label;
                crop.weighted_output_image(r, c) = weighted_label(label, weight);
            }
        }
    } else {
        crop.hr_output_image[0].set_size(lab_image.nr(), lab_image.nc());
        crop.hr_output_image[1].set_size(lab_image.nr(), lab_image.nc());
        for (long r = 0; r < lab_image.nr(); ++r) {
            for (long c = 0; c < lab_image.nc(); ++c) {
                crop.hr_output_image[0](r, c) = quantize_n_bits(lab_image(r, c).a, hr_res_nb_bits);
                crop.hr_output_image[1](r, c) = quantize_n_bits(lab_image(r, c).b, hr_res_nb_bits);
            }
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
    const double threshold = 3.27;
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
    long processed_files = 0;

    for (const auto& file : training_images) {
        try {
            matrix<rgb_pixel> input_image;
            load_image(input_image, file.full_name());            
            if (is_grayscale(input_image) || is_two_small(input_image)) {
                cout << "Too small or grayscale image, deleting: " << file.full_name() << endl;
                fs::remove(file.full_name()); // Remove small images
            }
            size_t prev_nc = input_image.nc(), prev_nr = input_image.nr();
            resize_max(input_image, 1024);
            if (input_image.nc() != prev_nc && input_image.nr() != prev_nr) {
                save_jpeg(input_image, file.full_name(), 95);
                cout << "Normalized: " << file.full_name() << endl; // Resize two large images
            }            
        } catch (...) {
            cerr << "Error processing file: " << file.full_name() << ", deleting" << endl;
            fs::remove(file.full_name()); // Remove bad images
        }
        if ((processed_files++ % 100) == 0) {
            long progress_percentage = static_cast<long>((static_cast<float>(processed_files) / total_files * 100));
            cout << "Normalization progress: [" << string(long(progress_percentage / 2), '=') << "] " << progress_percentage << "%" << "\r";
            cout.flush();
        }
        if (g_interrupted) break;
    }
    cout << endl;
}

// ----------------------------------------------------------------------------------------
std::vector<std::array<matrix<float>, 2>> get_generated_images(const tensor& out) {
    std::vector<std::array<matrix<float>, 2>> images;
    for (long n = 0; n < out.num_samples(); ++n) {
        std::array<matrix<float>, 2> output_image;
        output_image[0] = image_plane(out, n, 0);
        output_image[1] = image_plane(out, n, 1);
        images.push_back(output_image);
    }
    return images;
}
void norm_output_images(std::vector<std::array<matrix<float>, 2>>& output_images) {
    for (auto& output_image : output_images) {
        for (int k = 0; k < 2; ++k) {
            for (long r = 0; r < output_image[k].nr(); ++r) {
                for (long c = 0; c < output_image[k].nc(); ++c) {
                    if (output_image[k](r,c) < 0) output_image[k](r, c) = (0.0f - avg_hr_output_value) / max_hr_output_value;
                    else if (output_image[k](r, c) > max_hr_output_value) output_image[k](r, c) = (max_hr_output_value - avg_hr_output_value) / max_hr_output_value;
                    else output_image[k](r, c) = (output_image[k](r, c) - avg_hr_output_value) / max_hr_output_value;
                }
            }
        }
    }
}
// ----------------------------------------------------------------------------------------
bool is_directory(const std::string& path) {
    struct stat s;
    if (stat(path.c_str(), &s) == 0) {
        if (s.st_mode & S_IFDIR) {
            return true;
        }
    }
    return false;
}
void parse_directory(const std::string& path, std::vector<std::string>& files) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) files.push_back(entry.path().string());
    }
}

int main(int argc, char** argv) try {
    const long update_display = 30;
    const long max_minutes_elapsed = 8;
    bool do_augmentation = false, import_backbone = false, blur_channels = false, boost_colors = false;
    double initial_learning_rate = 1e-1;
    size_t minibatch_size = 20, patience = 10000;
    po::options_description desc("Program options");
    desc.add_options()
        ("normalization", po::value<string>(), "normalize images <dir>")
        ("classification", po::value<string>(), "train classification model <dir>")
        ("regression", po::value<string>(), "train regression model <dir>")
        ("regression-gan", po::value<string>(), "train regression model <dir>")
        ("classification-test", po::value<string>(), "test classification model <dir or file>")
        ("regression-test", po::value<string>(), "test regression model <dir or file>")
        ("export-backbone", po::value<string>(), "export backbone from a model <model name>")
        ("import-backbone", po::bool_switch(&import_backbone)->default_value(false), "import backbone to initiate a model")
        ("image-augmentation", po::bool_switch(&do_augmentation)->default_value(false), "do image augmentation during training")
        ("initial-learning-rate", po::value<double>(&initial_learning_rate)->default_value(1e-1), "set the initial learning rate (default 0.1)")
        ("minibatch-size", po::value<size_t>(&minibatch_size)->default_value(20), "set the minibatch size (default 20)")
        ("patience", po::value<size_t>(&patience)->default_value(10000), "set the patience parameter (default 10000)")
        ("low-bulk-convert", po::value<string>(), "convert multiple images at the same time <dir> (low resolution model)")
        ("high-bulk-convert", po::value<string>(), "convert multiple images at the same time <dir> (high resolution model)")
        ("blur-channels", po::bool_switch(&blur_channels)->default_value(false), "apply slight blur to the color channels")
        ("boost-colors", po::bool_switch(&boost_colors)->default_value(false), "enhance the vibrancy of colors")
        ("help", "this help message");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return EXIT_FAILURE;
    }
    std::srand(std::time(nullptr));
    dlib::rand rnd(std::rand());
    size_t iteration = 0;    
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
    set_dnn_prefer_smallest_algorithms();

    if (vm.count("normalization")) {
        normalize_images(vm["normalization"].as<string>());
    } else if (vm.count("export-backbone")) {
        const string input_model = vm["export-backbone"].as<string>();
        std::vector<string> model_names = { "lowres_colorify.dnn", "highres_colorify.dnn", "backbone_colorify.dnn" };
        net_type_lr net_lr;
        net_type_hr net_hr;
        net_backbone sub_net;
        cout << "Exporting the network backbone in progress... ";
        if (input_model.find(model_names[0]) != std::string::npos) {
            if (file_exists(input_model)) deserialize(input_model) >> net_lr;
            sub_net = layer<2>(net_lr);
        } else if (input_model.find(model_names[1]) != std::string::npos) {
            if (file_exists(input_model)) deserialize(input_model) >> net_hr;
            sub_net = layer<2>(net_hr);
        } else {
            cout << "You can only use <" << model_names[0] << "> or <" << model_names[1] << "> as input model" << endl;
            return EXIT_FAILURE;
        }
        serialize(model_names[2]) << sub_net;
        cout << "done" << endl;
    } else if (vm.count("classification") || vm.count("regression")) {
        const bool classification_training = vm.count("classification");
        const string input_dir = classification_training ? vm["classification"].as<string>() : vm["regression"].as<string>();
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(input_dir, dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (training_images.size() == 0) {
            cout << "Didn't find images for the training dataset" << endl;
            return EXIT_FAILURE;
        }
        current_state state;
        state.first_run_time = chrono::system_clock::now();
        state.last_run_time = state.first_run_time;

        // Instantiate the model        
        const string model_name = classification_training ? "lowres_colorify.dnn" : "highres_colorify.dnn";
        net_type_lr net_lr;
        net_type_hr net_hr;
        if (classification_training) {
            if (file_exists(model_name)) deserialize(model_name) >> net_lr;
            if (import_backbone && file_exists("backbone_colorify.dnn")) {
                cout << "Loading backbone... ";
                net_backbone sub_net;
                deserialize("backbone_colorify.dnn") >> sub_net;
                layer<2>(net_lr) = sub_net;
                cout << "done\n";
            }
        } else {
            if (file_exists(model_name)) deserialize(model_name) >> net_hr;
            if (import_backbone && file_exists("backbone_colorify.dnn")) {
                cout << "Loading backbone... ";
                net_backbone sub_net;
                deserialize("backbone_colorify.dnn") >> sub_net;
                layer<2>(net_hr) = sub_net;
                cout << "done\n";
            }
        }        

        const double min_learning_rate = 1e-5;
        const double weight_decay = 1e-4;
        const double momentum = 0.9;

        // Initialize the trainer
        dnn_trainer<net_type_lr> trainer_lr(net_lr, sgd(weight_decay, momentum));
        trainer_lr.set_learning_rate(initial_learning_rate);
        trainer_lr.set_learning_rate_shrink_factor(0.1);
        trainer_lr.set_mini_batch_size(minibatch_size);
        trainer_lr.set_iterations_without_progress_threshold(patience);
        trainer_lr.set_min_learning_rate(min_learning_rate);
        trainer_lr.be_verbose();        
        set_all_bn_running_stats_window_sizes(net_lr, 1000);
        // --
        dnn_trainer<net_type_hr> trainer_hr(net_hr, sgd(weight_decay, momentum));
        trainer_hr.set_learning_rate(initial_learning_rate);
        trainer_hr.set_learning_rate_shrink_factor(0.1);
        trainer_hr.set_mini_batch_size(minibatch_size);
        trainer_hr.set_iterations_without_progress_threshold(patience);
        trainer_hr.set_min_learning_rate(min_learning_rate);
        trainer_hr.be_verbose();
        set_all_bn_running_stats_window_sizes(net_hr, 1000);
        double cur_learning_rate = classification_training ? trainer_lr.get_learning_rate() : trainer_hr.get_learning_rate();
        disable_duplicative_biases(net_lr);
        disable_duplicative_biases(net_hr);

        // Output training parameters
        training_sample sample;
        const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
        matrix<rgb_pixel> input_image;
        load_image(input_image, image_info.full_name());
        randomly_crop_image(input_image, sample, rnd, classification_training, do_augmentation);
        if (classification_training) {
            net_lr(sample.input_image);
            cout << net_lr << endl;
            cout << "The network has " << net_lr.num_layers << " layers in it" << std::endl;
            cout << std::endl << trainer_lr << endl;
        } else {
            net_hr(sample.input_image);
            cout << net_hr << std::endl;
            cout << "The network has " << net_hr.num_layers << " layers in it" << std::endl;
            cout << std::endl << trainer_hr << endl;
        }        
        // Total images in the dataset
        cout << "images in dataset: " << training_images.size() << endl;

        // Use some threads to preload images
        dlib::pipe<training_sample> data(minibatch_size);
        auto f = [&data, &training_images, &classification_training, &do_augmentation](time_t seed) {
            dlib::rand rnd(time(nullptr) + seed);
            matrix<rgb_pixel> input_image;
            training_sample temp;
            while (data.is_enabled()) {
                const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
                try { 
                    load_image(input_image, image_info.full_name());
                    randomly_crop_image(input_image, temp, rnd, classification_training, do_augmentation);
                    data.enqueue(temp);
                } catch (...) {
                    cerr << "Error during image loading: " << image_info.full_name() << endl;
                }                
            }
        };
        std::thread data_loader1([f]() { f(1); });
        std::thread data_loader2([f]() { f(2); });
        std::thread data_loader3([f]() { f(3); });
        cout << "Waiting for the initial pipe loading... ";
        while (data.size() < minibatch_size) std::this_thread::sleep_for(std::chrono::seconds(1));
        cout << "done" << endl;

        std::vector<matrix<uint16_t>> lr_labels;
        std::vector<matrix<weighted_label_>> weighted_labels;
        std::vector<std::array<matrix<float>, 2>> hr_labels;
        std::vector<matrix<gray_pixel>> samples;
        dlib::image_window win;
        while (!g_interrupted && cur_learning_rate >= min_learning_rate) {
            // Train
            lr_labels.clear();
            weighted_labels.clear();
            hr_labels.clear();
            samples.clear();
            while (samples.size() < minibatch_size) {
                data.dequeue(sample);
                samples.push_back(sample.input_image);
                if (classification_training) {
                    lr_labels.push_back(sample.lr_output_image);
                    weighted_labels.push_back(sample.weighted_output_image);
                } else {
                    hr_labels.push_back(sample.hr_output_image);
                }
            }            
            if (++iteration % update_display == 0) { // We should see that the generated images start looking like samples
                std::vector<matrix<rgb_pixel>> disp_imgs;
                matrix<rgb_pixel> src_img, rgb_gen, gray_img, image_to_save;
                size_t pos_i = 0, max_iter = __min(samples.size(), 4);
                if (classification_training) {
                    trainer_lr.get_net(dlib::force_flush_to_disk::no);
                    auto gen_samples = net_lr(samples);                    
                    for (auto& image : gen_samples) {
                        assign_image(gray_img, samples[pos_i]);
                        src_img = concat_channels(samples[pos_i], lr_labels[pos_i]);
                        rgb_gen = concat_channels(samples[pos_i], image);
                        disp_imgs.push_back(join_rows(src_img, join_rows(gray_img, rgb_gen)));
                        if (++pos_i >= max_iter) break;
                    }
                } else {
                    trainer_hr.get_net(dlib::force_flush_to_disk::no);
                    auto gen_samples = net_hr(samples);                    
                    for (auto& image : gen_samples) {
                        assign_image(gray_img, samples[pos_i]);
                        src_img = concat_channels(samples[pos_i], hr_labels[pos_i]);
                        rgb_gen = concat_channels(samples[pos_i], image);
                        disp_imgs.push_back(join_rows(src_img, join_rows(gray_img, rgb_gen)));
                        if (++pos_i >= max_iter) break;
                    }
                }
                image_to_save = tile_images(disp_imgs);
                if (classification_training) save_jpeg(image_to_save, "lowres_model_training.jpg", 95);
                else save_jpeg(image_to_save, "highres_model_training.jpg", 95);
                win.set_image(image_to_save);
                win.set_title("COLORIFY - Supervised-learning process, step#: " + to_string(iteration) + " - " + to_string(max_iter) + " samples - Original | Grayscale | Colorized");
            }
            if (classification_training) trainer_lr.train_one_step(samples, weighted_labels);
            else trainer_hr.train_one_step(samples, hr_labels);
            cur_learning_rate = classification_training ? trainer_lr.get_learning_rate() : trainer_hr.get_learning_rate();
            // Check if the model has to be saved
            if (iteration % 200 == 0) {
                chrono::time_point<chrono::system_clock> current_time = chrono::system_clock::now();
                double minutes_elapsed = chrono::duration_cast<chrono::minutes>(current_time - state.last_run_time).count();                
                if (minutes_elapsed >= max_minutes_elapsed) {
                    if (classification_training) {
                        trainer_lr.get_net(dlib::force_flush_to_disk::no);
                        net_lr.clean();
                        serialize(model_name) << net_lr;
                    } else {
                        trainer_hr.get_net(dlib::force_flush_to_disk::no);
                        net_hr.clean();
                        serialize(model_name) << net_hr;
                    }
                    cout << "checkpoint#:\tModel <" << model_name << "> saved on disk" << endl;
                    state.last_run_time = current_time;
                }
            }
        }
        data.disable();
        data_loader1.join();
        data_loader2.join();
        data_loader3.join();

        // Once the training has finished, we don't need the discriminator any more. We just keep the generator
        // We also save the checkpoint again to iterate the learning process
        if (classification_training) {
            trainer_lr.get_net(dlib::force_flush_to_disk::no);
            net_lr.clean();
            serialize(model_name) << net_lr;
        } else {
            trainer_hr.get_net(dlib::force_flush_to_disk::no);
            net_hr.clean();
            serialize(model_name) << net_hr;
        }
    } else if (vm.count("regression-gan")) {
        const string input_dir = vm["regression-gan"].as<string>();
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(input_dir, dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (training_images.size() == 0) {
            cout << "Didn't find images for the training dataset" << endl;
            return EXIT_FAILURE;
        }
        current_state state;
        state.first_run_time = chrono::system_clock::now();
        state.last_run_time = state.first_run_time;

        // Instantiate the model        
        const string model_name = "highres_colorify.dnn";
        net_type_hr net_hr;
        net_discriminator net_d;
        visit_computational_layers(net_d, [](leaky_relu_& l) { l = leaky_relu_(0.2); });
        disable_duplicative_biases(net_d);

        // The solvers for the generator and discriminator networks
        const double weight_decay = 1e-4;
        const double momentum_1 = 0.5, momentum_2 = 0.999;
        std::vector<adam> g_solvers(net_hr.num_computational_layers, adam(weight_decay, momentum_1, momentum_2));
        std::vector<adam> d_solvers(net_d.num_computational_layers, adam(weight_decay, momentum_1, momentum_2));
        double learning_rate = 2e-4;

        // Resume training from last model file (only for the generator)
        if (file_exists(model_name)) deserialize(model_name) >> net_hr;
        if (import_backbone && file_exists("backbone_colorify.dnn")) {
            cout << "Loading backbone... ";
            net_backbone sub_net;
            deserialize("backbone_colorify.dnn") >> sub_net;
            layer<2>(net_hr) = sub_net;
            cout << "done\n";
        }

        // Total images in the dataset
        cout << "images in dataset: " << training_images.size() << endl;

        // Show networks
        training_sample sample;
        matrix<rgb_pixel> input_image;
        const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
        load_image(input_image, image_info.full_name());
        randomly_crop_image(input_image, sample, rnd, false, do_augmentation);
        net_hr(sample.input_image);
        net_d(sample.hr_output_image);
        cout << "generator (" << count_parameters(net_hr) << " parameters)" << endl;
        cout << net_hr << endl;
        cout << "discriminator (" << count_parameters(net_d) << " parameters)" << endl;
        cout << net_d << endl;

        // Use some threads to preload images
        dlib::pipe<training_sample> data(minibatch_size);
        auto f = [&data, &training_images, &do_augmentation](time_t seed) {
            dlib::rand rnd(time(nullptr) + seed);
            matrix<rgb_pixel> input_image;
            training_sample temp;
            while (data.is_enabled()) {
                const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
                try {
                    load_image(input_image, image_info.full_name());
                    randomly_crop_image(input_image, temp, rnd, false, do_augmentation);
                    data.enqueue(temp);
                } catch (...) {
                    cerr << "Error during image loading: " << image_info.full_name() << endl;
                }
            }
        };
        std::thread data_loader1([f]() { f(1); });
        std::thread data_loader2([f]() { f(2); });
        cout << "Waiting for the initial pipe loading... ";
        while (data.size() < minibatch_size) std::this_thread::sleep_for(std::chrono::seconds(1));
        cout << "done" << endl;

        const std::vector<float> real_labels(minibatch_size, 1), gen_labels(minibatch_size, -1);
        resizable_tensor real_samples_tensor, gen_samples_tensor, grays_tensor;
        running_stats<double> g_loss, d_loss;
        dlib::image_window win;
        std::vector<std::array<matrix<float>, 2>> output_samples;
        std::vector<matrix<gray_pixel>> gray_samples;
        while (!g_interrupted) {
            output_samples.clear();
            gray_samples.clear();
            while (output_samples.size() < minibatch_size) {
                data.dequeue(sample);
                output_samples.push_back(sample.hr_output_image);
                gray_samples.push_back(sample.input_image);
            }            

            // Train the discriminator with real images
            //norm_output_images(output_samples);
            net_d.to_tensor(output_samples.begin(), output_samples.end(), real_samples_tensor);
            net_d.forward(real_samples_tensor);
            d_loss.add(net_d.compute_loss(real_samples_tensor, real_labels.begin()));
            net_d.back_propagate_error(real_samples_tensor);
            net_d.update_parameters(d_solvers, learning_rate);

            // Train the discriminator with fake images
            net_hr.to_tensor(gray_samples.begin(), gray_samples.end(), grays_tensor);
            net_hr.forward(grays_tensor);
            auto gen_samples = get_generated_images(layer<1>(net_hr).get_output());
            //norm_output_images(gen_samples);
            net_d.to_tensor(gen_samples.begin(), gen_samples.end(), gen_samples_tensor);
            net_d.forward(gen_samples_tensor);
            d_loss.add(net_d.compute_loss(gen_samples_tensor, gen_labels.begin()));
            net_d.back_propagate_error(gen_samples_tensor);
            net_d.update_parameters(d_solvers, learning_rate);

            // Forward the fake samples and compute the loss with real labels
            g_loss.add(net_d.compute_loss(gen_samples_tensor, real_labels.begin()));
            net_d.back_propagate_error(gen_samples_tensor);
            resizable_tensor d_grad = net_d.get_final_data_gradient();
            
            /* {
                resizable_tensor original_tensor = d_grad;
                d_grad.set_size(d_grad.num_samples(), 2, d_grad.nr(), d_grad.nc());
                const float* const out_data = original_tensor.host();
                float* const in_data = d_grad.host();

                for (int ii = 0; ii < d_grad.num_samples(); ++ii) {
                    for (int jj = 0; jj < d_grad.nr(); ++jj) {
                        for (int kk = 0; kk < d_grad.nc(); ++kk) {
                            float sum = 0.0f;
                            for (int k = 0; k < original_tensor.k(); ++k) {
                                const size_t index = ((ii * original_tensor.k() + k) * original_tensor.nr() + jj) * original_tensor.nc() + kk;
                                if (index < original_tensor.size()) sum += out_data[index];
                            }
                            for (int k = 0; k < 2; ++k) {
                                const size_t index = ((ii * d_grad.k() + k) * d_grad.nr() + jj) * d_grad.nc() + kk;
                                if (index < d_grad.size()) in_data[index] = sum / original_tensor.k();
                            }
                        }
                    }
                }
            }*/
            net_hr.back_propagate_error(grays_tensor, d_grad);
            net_hr.update_parameters(g_solvers, learning_rate);

            // See that the generated images start looking like samples
            if (++iteration % update_display == 0) { // Display
                std::vector<matrix<rgb_pixel>> disp_imgs;
                matrix<rgb_pixel> rgb_src, rgb_gen, gray_img;
                size_t pos_i = 0, max_iter = __min(gen_samples.size(), 4);
                for (auto& image : gen_samples) {
                    /*for (int c = 0; c < 2; ++c) {
                        output_samples[pos_i][c] = (output_samples[pos_i][c] * max_hr_output_value) + avg_hr_output_value;
                        gen_samples[pos_i][c] = (gen_samples[pos_i][c] * max_hr_output_value) + avg_hr_output_value;
                    }*/
                    assign_image(gray_img, gray_samples[pos_i]);
                    rgb_src = concat_channels(gray_samples[pos_i], output_samples[pos_i]);
                    rgb_gen = concat_channels(gray_samples[pos_i], gen_samples[pos_i]);
                    disp_imgs.push_back(join_rows(rgb_src, join_rows(gray_img, rgb_gen)));
                    if (++pos_i >= max_iter) break;
                }
                win.set_image(tile_images(disp_imgs));
                win.set_title("COLORIFY - GAN-learning process, step#: " + to_string(iteration) + " - " + to_string(max_iter) + " samples - Original | Grayscale | Colorized");
            }
            if (iteration % 100 == 0) { // Standard progress
                cout << "step#: " << iteration << "\tdiscriminator loss: " << d_loss.mean() * 2 <<
                    "\tgenerator loss: " << g_loss.mean() << endl;
                double d_loss_mean = d_loss.mean(), g_loss_mean = g_loss.mean();
                d_loss.clear();
                g_loss.clear();
                d_loss.add(d_loss_mean);
                g_loss.add(g_loss_mean);

            }
            // Check if the model has to be saved
            if (iteration % 200 == 0) {
                chrono::time_point<chrono::system_clock> current_time = chrono::system_clock::now();
                double minutes_elapsed = chrono::duration_cast<chrono::minutes>(current_time - state.last_run_time).count();
                if (minutes_elapsed >= max_minutes_elapsed) {
                    net_hr.clean();
                    serialize(model_name) << net_hr;                   
                    cout << "checkpoint#:\tModel <" << model_name << "> saved on disk" << endl;
                    state.last_run_time = current_time;
                }
            }
        }
        data.disable();
        data_loader1.join();
        data_loader2.join();

        // Once the training has finished, we don't need the discriminator any more. We just keep the generator
        // We also save the checkpoint again to iterate the learning process
        net_hr.clean();
        serialize(model_name) << net_hr;
    } else if (vm.count("classification-test") || vm.count("regression-test")) {
        bool use_lr_model = vm.count("classification-test");
        const string input_dir = use_lr_model ? vm["classification-test"].as<string>() : vm["regression-test"].as<string>();
        std::vector<string> images;
        if (!is_directory(input_dir)) {
            images.push_back(input_dir);
        } else {
            parse_directory(input_dir, images);
        }
        cout << "total images to colorify: " << images.size() << endl;
        if (images.size() == 0) {
            cout << "Didn't find images to colorify" << endl;
            return EXIT_FAILURE;
        }

        // Load the mode
        const string model_name = use_lr_model ? "lowres_colorify.dnn" : "highres_colorify.dnn";
        net_type_lr net_lr;
        net_type_hr net_hr;
        if (use_lr_model) {
            if (file_exists(model_name)) deserialize(model_name) >> net_lr;
            else {
                cout << "Didn't find the precomputed model: " << model_name << endl;
                return EXIT_FAILURE;
            }
        } else {
            if (file_exists(model_name)) deserialize(model_name) >> net_hr;
            else {
                cout << "Didn't find the precomputed model: " << model_name << endl;
                return EXIT_FAILURE;
            }
        }

        dlib::image_window win;
        matrix<rgb_pixel> input_image, rgb_image, blur_image, display_gray_image;
        matrix<gray_pixel> gray_image, temp_gray_image;
        matrix<lab_pixel> lab_image;
        for (auto& i : images) {            
            try { load_image(input_image, i); }
            catch (...) {
                cerr << "Error during image loading: " << i << endl;
                continue;
            }
            if (is_two_small(input_image)) continue;
            const bool is_grayscale_image = is_grayscale(input_image);
            resize_max(input_image, std_image_size * 1.5);
            rgb_image_to_grayscale_image(input_image, gray_image);
            assign_image(display_gray_image, gray_image);         
            // --- Core process for colorization
            {
                assign_image(temp_gray_image, gray_image);
                resize_inplace(temp_gray_image, std_image_size);
                if (use_lr_model) {
                    matrix<uint16_t> output = net_lr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                } else {
                    std::array<matrix<float>, 2> output = net_hr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                }                
                scale_image(gray_image.nr(), gray_image.nc(), rgb_image);
                if (blur_channels) {
                    gaussian_blur(rgb_image, blur_image, 0.8);
                    assign_image(lab_image, blur_image);
                } else {
                    assign_image(lab_image, rgb_image);
                }
                for (long r = 0; r < lab_image.nr(); ++r)
                    for (long c = 0; c < lab_image.nc(); ++c)
                        lab_image(r, c).l = gray_image(r, c);
                assign_image(rgb_image, lab_image);
                if (boost_colors) {
                    const float saturation_boost = 0.17f;
                    matrix<hsi_pixel> hsi_image;
                    assign_image(hsi_image, rgb_image);
                    for (long r = 0; r < hsi_image.nr(); ++r) {
                        for (long c = 0; c < hsi_image.nc(); ++c) {
                            hsi_image(r, c).s = __min(255, std::round(static_cast<float>(hsi_image(r, c).s) * (1 + saturation_boost)));
                        }
                    }
                    assign_image(rgb_image, hsi_image);
                }
            }
            // ---
            if (is_grayscale_image) {
                win.set_title("COLORIFY - Original (grayscale) " + to_string(input_image.nc()) + "x" + to_string(input_image.nr()) + ") | Generated (" + to_string(rgb_image.nc()) + "x" + to_string(rgb_image.nr()) + ")");
                win.set_image(join_rows(input_image, rgb_image));
            } else {
                win.set_title("COLORIFY - Original " + to_string(input_image.nc()) + "x" + to_string(input_image.nr()) + ") | Grayscale | Generated (" + to_string(rgb_image.nc()) + "x" + to_string(rgb_image.nr()) + ")");
                win.set_image(join_rows(input_image, join_rows(display_gray_image, rgb_image)));
            }
            cout << i << " - Hit enter to process the next image or 'q' to quit";
            char c = std::cin.get();
            if (c == 'q' || c == 'Q') break;
        }
    } else if (vm.count("low-bulk-convert") || vm.count("high-bulk-convert")) {
        bool use_lr_model = vm.count("low-bulk-convert");
        const string input_dir = use_lr_model ? vm["low-bulk-convert"].as<string>() : vm["high-bulk-convert"].as<string>();
        if (!is_directory(input_dir)) {
            cout << "<" << input_dir << "> isn't a directory" << endl;
            return EXIT_FAILURE;
        }
        const std::vector<file> images = dlib::get_files_in_directory_tree(input_dir, dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (images.size() == 0) {
            cout << "Didn't find images for the bulk colorization process" << endl;
            return EXIT_FAILURE;
        }
        cout << "total images to colorify: " << images.size() << endl;
        
        // Load the mode
        const string model_name = use_lr_model ? "lowres_colorify.dnn" : "highres_colorify.dnn";
        net_type_lr net_lr;
        net_type_hr net_hr;
        if (use_lr_model) {
            if (file_exists(model_name)) deserialize(model_name) >> net_lr;
            else {
                cout << "Didn't find the precomputed model: " << model_name << endl;
                return EXIT_FAILURE;
            }
        } else {
            if (file_exists(model_name)) deserialize(model_name) >> net_hr;
            else {
                cout << "Didn't find the precomputed model: " << model_name << endl;
                return EXIT_FAILURE;
            }
        }
        matrix<rgb_pixel> input_image, rgb_image, blur_image;
        matrix<gray_pixel> gray_image, temp_gray_image;
        matrix<lab_pixel> lab_image;
        size_t count = 0;
        cout << "please wait, bulk conversion in progress... [" << count  << "/" << images.size() << "]\r";
        for (auto& i : images) {
            try { load_image(input_image, i.full_name()); }
            catch (...) {
                cerr << "Error during image loading: " << i << endl;
                continue;
            }
            rgb_image_to_grayscale_image(input_image, gray_image);
            // --- Core process for colorization
            {
                assign_image(temp_gray_image, gray_image);
                resize_inplace(temp_gray_image, std_image_size);
                if (use_lr_model) {
                    matrix<uint16_t> output = net_lr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                } else {
                    std::array<matrix<float>, 2> output = net_hr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                }
                scale_image(gray_image.nr(), gray_image.nc(), rgb_image);
                if (blur_channels) {
                    gaussian_blur(rgb_image, blur_image, 0.8);
                    assign_image(lab_image, blur_image);
                } else {
                    assign_image(lab_image, rgb_image);
                }
                for (long r = 0; r < lab_image.nr(); ++r)
                    for (long c = 0; c < lab_image.nc(); ++c)
                        lab_image(r, c).l = gray_image(r, c);
                assign_image(rgb_image, lab_image);
                if (boost_colors) {
                    const float saturation_boost = 0.17f;
                    matrix<hsi_pixel> hsi_image;
                    assign_image(hsi_image, rgb_image);
                    for (long r = 0; r < hsi_image.nr(); ++r) {
                        for (long c = 0; c < hsi_image.nc(); ++c) {
                            hsi_image(r, c).s = __min(255, std::round(static_cast<float>(hsi_image(r, c).s) * (1 + saturation_boost)));
                        }
                    }
                    assign_image(rgb_image, hsi_image);
                }
            }
            // ---
            const string suffix = "_colorized";
            fs::path filePath(i.full_name()), stem = filePath.stem(), extension = filePath.extension();
            fs::path output_finalement = (filePath.parent_path() / stem).string() + suffix + extension.string();
            save_jpeg(rgb_image, output_finalement.string(), 95);
            cout << "please wait, bulk conversion in progress... [" << ++count << "/" << images.size() << "]\r";
            if (g_interrupted) break;
        }
        cout << "please wait, bulk conversion in progress... [" << images.size() << "/" << images.size() << "]\ndone" << endl;
    }
} catch (std::exception& e) {
    cout << e.what() << endl;
}