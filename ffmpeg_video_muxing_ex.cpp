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
template <typename AbImageType>
matrix<rgb_pixel> concat_channels(const matrix<gray_pixel>& gray_image, const AbImageType& ab_image) {
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
template <typename pixel_type>
void scale_image(long nr, long nc, matrix<pixel_type>& dst) {
    matrix<pixel_type> resized(nr, nc);
    resize_image(dst, resized, interpolate_bilinear());
    assign_image(dst, resized);
}

// ----------------------------------------------------------------------------------------
template <typename GeneratorType>
matrix<rgb_pixel> generate_image(GeneratorType& net, const matrix<gray_pixel>& src) {
    static_assert(std::is_same_v<GeneratorType, lr_generator_type> || std::is_same_v<GeneratorType, hr_generator_type>,
        "net must be either lr_generator_type or hr_generator_type");

    matrix<gray_pixel> gray_image = src;
    resize_inplace(gray_image, std_image_size);
    matrix<rgb_pixel> rgb_image, blur_image;
    if constexpr (std::is_same_v<GeneratorType, lr_generator_type>) {
        matrix<uint16_t> output = net(gray_image);
        rgb_image = concat_channels(gray_image, output);
        scale_image(src.nr(), src.nc(), rgb_image);
    }
    else if constexpr (std::is_same_v<GeneratorType, hr_generator_type>) {
        std::array<matrix<float>, 2> output = net(gray_image);
        rgb_image = concat_channels(gray_image, output);
    }
    gaussian_blur(rgb_image, blur_image);
    scale_image(src.nr(), src.nc(), blur_image);
    matrix<lab_pixel> lab_image;
    assign_image(lab_image, blur_image);
    for (long r = 0; r < lab_image.nr(); ++r)
        for (long c = 0; c < lab_image.nc(); ++c)
            lab_image(r, c).l = src(r, c);
    assign_image(rgb_image, lab_image);
    return rgb_image;
}

int main(const int argc, const char** argv)
try {
    command_line_parser parser;
    parser.add_option("in", "input video", 1);
    parser.add_option("out", "output file", 1);
    parser.set_group_name("Model type used for colorization");
    parser.add_option("low-resolution", "using the precomputed indexed \"a*b channels\" model");
    parser.add_option("high-resolution", "using the high-definition 65k colors model (default model)");
    parser.set_group_name("Miscellaneous options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.add_option("height", "height of encoded stream (defaults to whatever is in the video file)", 1);
    parser.add_option("width", "width of encoded stream (defaults to whatever is in the video file)", 1);
    parser.add_option("video-codec", "video codec name (e.g. \"h264\")", 1);
    parser.add_option("audio-codec", "audio codec name (e.g. \"aac\")", 1);

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"in", "out", "low-resolution", "high-resolution", "video-codec", "audio-codec", "height", "width" };
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
    bool high_resolution_model = parser.option("low-resolution") ? false : true;
    lr_generator_type lr_generator;
    hr_generator_type hr_generator;
    if (high_resolution_model) {
        if (file_exists("highres_colorify.dnn")) {
            deserialize("highres_colorify.dnn") >> hr_generator;
        } else {
            cout << "Didn't find the model (highres_colorify.dnn)" << endl;
            cout << "This model can be downloaded here: https://github.com/Cydral/Colorify/highres_colorify.zip\n";
            return EXIT_FAILURE;
        }
    } else {
        if (file_exists("lowres_colorify.dnn")) {
            deserialize("lowres_colorify.dnn") >> lr_generator;
        } else {
            cout << "Didn't find the model (lowres_colorify.dnn)" << endl;
            cout << "This model can be downloaded here: https://github.com/Cydral/Colorify/lowres_colorify.zip\n";
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

    muxer writer([&] {
        muxer::args args;
        args.filepath     = output_filepath;
        args.enable_image = cap.video_enabled();
        args.enable_audio = cap.audio_enabled();
        if (args.enable_image) {
            args.args_image.codec_name  = get_option(parser, "video-codec", "mpeg4");
            args.args_image.h           = get_option(parser, "height", cap.height());
            args.args_image.w           = get_option(parser, "width",  cap.width());
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
    cout << "Source Video codec: " << cap.get_video_codec_name() << " (" << cap.width() << "x" << cap.height() << ") => " << writer.get_video_codec_name() << " (" << writer.width() << "x" << writer.height() << ")" << endl;
    cout << "Frame rate: " << cap.fps() << " fps" << endl;

    frame f;
    matrix<gray_pixel> gray_image;
    matrix<rgb_pixel> input_image, gen_image;
    uint64_t processed_samples = 0;

    dlib::image_window win;
    win.set_title("COLORIFY: <" + output_filepath + ">");
    const resizing_args args_image{ 0, 0, pix_traits<rgb_pixel>::fmt };
    cout << "Colorizing the video on progress\nPlease wait or press CTRL+C to stop" << endl;
    while (cap.read(f, args_image) && !g_interrupted) {
        if (f.is_image()) {
            convert(f, input_image);
            rgb_image_to_grayscale_image(input_image, gray_image);
            gen_image = high_resolution_model ? generate_image(hr_generator, gray_image) : generate_image(lr_generator, gray_image);
            convert(gen_image, f);
            resize_inplace(input_image, std_image_size);
            resize_inplace(gen_image, std_image_size);
            win.set_image(join_rows(input_image, gen_image));          
        }
        writer.push(std::move(f));
        if ((++processed_samples % 100) == 0) {
            long progress_percentage = static_cast<long>((static_cast<double>(processed_samples) / cap.estimated_nframes() * 100.0f));
            std::cout << "Colorization progress: [" << string(long(progress_percentage / 2), '=') << "] " << progress_percentage << "%" << "\r";
            std::cout.flush();
        }
    }
    cout << "Conversion done, flushing on disk... ";
    writer.flush();
    cout << "done" << endl;

    return EXIT_SUCCESS;
}
catch (const std::exception& e) {
    cout << e.what() << '\n';
    return EXIT_FAILURE;
}
