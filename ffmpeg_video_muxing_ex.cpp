#include <cstdio>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/gui_widgets.h>
#include "Model.h"

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

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
void resize_inplace(matrix<pixel_type>& inout, long nc, long nr) {
    if (inout.nr() != nr || inout.nc() != nc) {
        matrix<pixel_type> mem_img;
        mem_img.set_size(nc, nr);
        resize_image(inout, mem_img);
        inout = mem_img;
    }
}

// ----------------------------------------------------------------------------------------
const uint8_t hr_res_nb_bits = 5;
template <typename AbImageType>
matrix<rgb_pixel> concat_channels(const matrix<gray_pixel>& gray_image, const AbImageType& ab_image) {
    matrix<lab_pixel> lab_image(gray_image.nr(), gray_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            lab_image(r, c).l = gray_image(r, c);
            if constexpr (std::is_same<AbImageType, matrix<uint16_t>>::value) {
                lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) >> 4, 4));
                lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(ab_image(r, c) & 0xF, 4));
            }
            else if constexpr (std::is_same<AbImageType, std::array<matrix<float>, 2>>::value) {
                lab_image(r, c).a = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[0](r, c)), hr_res_nb_bits));
                lab_image(r, c).b = static_cast<uint8_t>(dequantize_n_bits(std::round(ab_image[1](r, c)), hr_res_nb_bits));
            }
        }
    }
    matrix<rgb_pixel> output;
    assign_image(output, lab_image);
    return output;
}

// ----------------------------------------------------------------------------------------
template <typename pixel_type>
void scale_image(long nr, long nc, matrix<pixel_type>& dst) {
    matrix<pixel_type> resized(nr, nc);
    resize_image(dst, resized, interpolate_bilinear());
    assign_image(dst, resized);
}

int main(const int argc, const char** argv)
try {
    command_line_parser parser;
    parser.add_option("in", "input video", 1);
    parser.add_option("out", "output file", 1);
    parser.set_group_name("Model type used for colorization");
    parser.add_option("low-resolution", "using the precomputed indexed \"a*b channels\" model");
    parser.add_option("high-resolution", "using the high-definition 65k colors model (default model)");
    parser.add_option("color-blurring", "appling a slight blur to the color channels excluding luminance");
    parser.add_option("color-boosting", "enhancing the vibrancy of colors for each frame");
    parser.set_group_name("Miscellaneous options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.add_option("height", "height of encoded stream (defaults to whatever is in the video file)", 1);
    parser.add_option("width", "width of encoded stream (defaults to whatever is in the video file)", 1);
    parser.add_option("dual-view", "side-by-side visualization of source and colorized images in the output");
    parser.add_option("video-codec", "video codec name (e.g. \"mpeg4\")", 1);
    parser.add_option("audio-codec", "audio codec name (e.g. \"aac\")", 1);

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"in", "out", "low-resolution", "high-resolution", "color-blurring", "color-boosting", "video-codec", "audio-codec", "height", "width" };
    parser.check_one_time_options(one_time_opts);

    parser.check_option_arg_range("width", 320, 7680);
    parser.check_option_arg_range("height", 240, 4320);

    if (parser.option("h") || parser.option("help")) {
        cout << "Usage: ffmpeg_video_muxing_ex --in input_video --out output_video\n";
        cout << "The different models can be downloaded from this address: https://github.com/Cydral/Colorify\n\n";
        parser.print_options();
        return 0;
    }
    if (!parser.option("in") || !parser.option("out")) {
        cout << "The input and output file names are required to initiate the colorization process\n";
        cout << "Run the program with the \"--help\" option for more information\n";
        return 0;
    }
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    // Check which DNN model to use for the colorization process and load the model
    const bool high_resolution_model = parser.option("low-resolution") ? false : true;
    const bool color_blurring = parser.option("color-blurring");
    const bool color_boosting = parser.option("color-boosting");
    const bool dual_view = parser.option("dual-view");
    const string model_name = high_resolution_model ? "highres_colorify.dnn" : "lowres_colorify.dnn";
    using net_type_lr = loss_multiclass_log_per_pixel_weighted<cont<256, 1, 1, 1, 1, generator_backbone<input<matrix<gray_pixel>>>>>;
    using net_type_hr = loss_mean_squared_per_channel_and_pixel<2, cont<2, 1, 1, 1, 1, generator_backbone<input<matrix<gray_pixel>>>>>;
    net_type_lr net_lr;
    net_type_hr net_hr;
    if (high_resolution_model) {
        if (file_exists(model_name)) deserialize(model_name) >> net_hr;
        else {
            cout << "Didn't find the model (" << model_name << ")" << endl;
            cout << "This model can be downloaded here: https://github.com/Cydral/Colorify/highres_colorify.bz2\n";
            return EXIT_FAILURE;
        }
    } else {
        if (file_exists(model_name)) deserialize(model_name) >> net_lr;
        else {
            cout << "Didn't find the model (" << model_name << ")" << endl;
            cout << "This model can be downloaded here: https://github.com/Cydral/Colorify/lowres_colorify.bz2\n";
            return EXIT_FAILURE;
        }
    }
    
    const std::string input_filepath  = parser.option("in").argument();
    const std::string output_filepath = parser.option("out").argument();
    demuxer cap(input_filepath);    

    if (!cap.is_open()) {
        cout << "Failed to open " << input_filepath << endl;
        return EXIT_FAILURE;
    }

    int output_width = get_option(parser, "width", cap.width());
    int output_height = get_option(parser, "height", cap.height());
    muxer writer([&] {
        muxer::args args;
        args.filepath     = output_filepath;
        args.enable_image = cap.video_enabled();
        args.enable_audio = cap.audio_enabled();
        if (args.enable_image) {
            args.args_image.codec_name  = get_option(parser, "video-codec", "mpeg4");
            args.args_image.h           = output_height;
            args.args_image.w           = output_width * (dual_view ? 2 : 1);
            args.args_image.fmt         = cap.pixel_fmt();
            args.args_image.framerate   = cap.fps();
        }
        if (args.enable_audio) {
            args.args_audio.codec_name      = get_option(parser, "audio-codec", cap.get_audio_codec_name());
            args.args_audio.sample_rate     = cap.sample_rate();
            args.args_audio.channel_layout  = cap.channel_layout();
            args.args_audio.fmt             = cap.sample_fmt();
        }
        return args;
    }());

    if (!writer.is_open()) {
        cout << "Failed to open " << output_filepath << endl;
        return EXIT_FAILURE;
    }
    // Display some information    
    cout << "Source Video codec: " << cap.get_video_codec_name() << " (" << cap.width() << "x" << cap.height() << ") => " << writer.get_video_codec_name() << " (" << output_width << "x" << output_height << ")" << endl;
    cout << "Frame rate: " << cap.fps() << " fps" << endl;

    frame f;
    matrix<gray_pixel> gray_image, temp_gray_image;
    matrix<rgb_pixel> input_image, rgb_image, dual_rgb_image, blur_image;
    matrix<lab_pixel> lab_image;
    uint64_t processed_samples = 0;

    dlib::image_window win;
    win.set_title("COLORIFY: <" + output_filepath + ">");
    const resizing_args args_image{ 0, 0, pix_traits<rgb_pixel>::fmt };
    cout << "Colorizing the video on progress\nPlease wait or press CTRL+C to stop" << endl;
    while (cap.read(f, args_image) && !g_interrupted) {
        if (f.is_image()) {
            convert(f, input_image);
            resize_inplace(input_image, output_width, output_height);
            rgb_image_to_grayscale_image(input_image, gray_image);
            // ---
            {
                assign_image(temp_gray_image, gray_image);
                resize_inplace(temp_gray_image, std_image_size);
                if (high_resolution_model) {
                    std::array<matrix<float>, 2> output = net_hr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                } else {
                    matrix<uint16_t> output = net_lr(temp_gray_image);
                    rgb_image = concat_channels(temp_gray_image, output);
                }                
                scale_image(gray_image.nr(), gray_image.nc(), rgb_image);
                if (color_blurring) {
                    gaussian_blur(rgb_image, blur_image, 0.8);
                    assign_image(lab_image, blur_image);
                } else {
                    assign_image(lab_image, rgb_image);
                }
                for (long r = 0; r < lab_image.nr(); ++r)
                    for (long c = 0; c < lab_image.nc(); ++c)
                        lab_image(r, c).l = gray_image(r, c);
                assign_image(rgb_image, lab_image);
                if (color_boosting) {
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
                if (dual_view) dual_rgb_image = join_rows(input_image, rgb_image);
            }
            // ---
            if (dual_view) convert(dual_rgb_image, f);
            else convert(rgb_image, f);
            resize_inplace(input_image, std_image_size);
            resize_inplace(rgb_image, std_image_size);
            win.set_image(join_rows(input_image, rgb_image));
        }
        writer.push(std::move(f));
        if ((++processed_samples % 100) == 0) {
            long progress_percentage = static_cast<long>((static_cast<double>(processed_samples) / (cap.estimated_nframes() * 3) * 100.0f));
            std::cout << "Colorization progress: [" << string(long(progress_percentage / 2), '=') << "] " << progress_percentage << "%" << "\r";
            std::cout.flush();
        }
    }
    std::cout << "Colorization progress: [" << string(long(100 / 2), '=') << "] " << 100 << "%" << "\n";
    cout << "Conversion done, flushing on disk... ";
    writer.flush();
    cout << "done" << endl;

    return EXIT_SUCCESS;
}
catch (const std::exception& e) {
    cout << e.what() << '\n';
    return EXIT_FAILURE;
}
