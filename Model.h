#ifndef DNN_MODEL_H
#define DNN_MODEL_H

// Inclusion de biblioth�ques n�cessaires
#include <iostream>
#include <string>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/statistics.h>
#include <dlib/dir_nav.h>

using namespace std;
using namespace dlib;

using gray_pixel = uint8_t;
using rgb565_pixel = uint16_t;
const size_t std_image_size = 227;

template<typename T1, typename T2>
constexpr auto uint8_to_uint16(T1 high, T2  low) { return (((static_cast<uint16_t>(high)) << 8) | (static_cast<uint16_t>(low))); }

// Introduce the building blocks used to define the U-Net network
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using blockt = BN<dlib::cont<N, 3, 3, 1, 1, dlib::relu<BN<dlib::cont<N, 3, 3, stride, stride, SUBNET>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N, 2, 2, 2, 2, dlib::skip1<dlib::tag2<blockt<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res = dlib::relu<residual<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_down = dlib::relu<residual_down<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using res_up = dlib::relu<residual_up<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares_up = dlib::relu<residual_up<block, N, dlib::affine, SUBNET>>;

// ----------------------------------------------------------------------------------------
template <typename SUBNET> using res64 = res<64, SUBNET>;
template <typename SUBNET> using res128 = res<128, SUBNET>;
template <typename SUBNET> using res256 = res<256, SUBNET>;
template <typename SUBNET> using res512 = res<512, SUBNET>;
template <typename SUBNET> using ares64 = ares<64, SUBNET>;
template <typename SUBNET> using ares128 = ares<128, SUBNET>;
template <typename SUBNET> using ares256 = ares<256, SUBNET>;
template <typename SUBNET> using ares512 = ares<512, SUBNET>;

template <typename SUBNET> using level1 = dlib::repeat<2, res64, res<64, SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<2, res128, res_down<128, SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<2, res256, res_down<256, SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2, res512, res_down<512, SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<2, ares64, ares<64, SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<2, ares128, ares_down<128, SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<2, ares256, ares_down<256, SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<2, ares512, ares_down<512, SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<2, res64, res_up<64, SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<2, res128, res_up<128, SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<2, res256, res_up<256, SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2, res512, res_up<512, SUBNET>>;

template <typename SUBNET> using alevel1t = dlib::repeat<2, ares64, ares_up<64, SUBNET>>;
template <typename SUBNET> using alevel2t = dlib::repeat<2, ares128, ares_up<128, SUBNET>>;
template <typename SUBNET> using alevel3t = dlib::repeat<2, ares256, ares_up<256, SUBNET>>;
template <typename SUBNET> using alevel4t = dlib::repeat<2, ares512, ares_up<512, SUBNET>>;

// ----------------------------------------------------------------------------------------
template <
    template<typename> class TAGGED,
    template<typename> class PREV_RESIZED,
    typename SUBNET
>
using resize_and_concat = dlib::add_layer<
    dlib::concat_<TAGGED, PREV_RESIZED>,
    PREV_RESIZED<dlib::resize_prev_to_tagged<TAGGED, SUBNET>>>;

template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100 + 1, SUBNET>;
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100 + 2, SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100 + 3, SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100 + 4, SUBNET>;

template <typename SUBNET> using utag1_ = dlib::add_tag_layer<2110 + 1, SUBNET>;
template <typename SUBNET> using utag2_ = dlib::add_tag_layer<2110 + 2, SUBNET>;
template <typename SUBNET> using utag3_ = dlib::add_tag_layer<2110 + 3, SUBNET>;
template <typename SUBNET> using utag4_ = dlib::add_tag_layer<2110 + 4, SUBNET>;

template <typename SUBNET> using concat_utag1 = resize_and_concat<utag1, utag1_, SUBNET>;
template <typename SUBNET> using concat_utag2 = resize_and_concat<utag2, utag2_, SUBNET>;
template <typename SUBNET> using concat_utag3 = resize_and_concat<utag3, utag3_, SUBNET>;
template <typename SUBNET> using concat_utag4 = resize_and_concat<utag4, utag4_, SUBNET>;

template <typename SUBNET> using dtag = dlib::add_tag_layer<3100 + 1, SUBNET>;

// ----------------------------------------------------------------------------------------
template <typename SUBNET> using generator_backbone =
    relu<bn_con<cont<64, 7, 7, 2, 2,
    concat_utag1<level1t<
    concat_utag2<level2t<
    concat_utag3<level3t<
    concat_utag4<level4t<
    level4<utag4<
    level3<utag3<
    level2<utag2<
    level1<max_pool<3, 3, 2, 2, utag1<
    relu<bn_con<con<64, 7, 7, 2, 2, SUBNET>>>>>>>>>>>>>>>>>>>>>>>;
template <typename SUBNET> using display_backbone =
    relu<affine<cont<64, 7, 7, 2, 2,
    concat_utag1<alevel1t<
    concat_utag2<alevel2t<
    concat_utag3<alevel3t<
    concat_utag4<alevel4t<
    alevel4<utag4<
    alevel3<utag3<
    alevel2<utag2<
    alevel1<max_pool<3, 3, 2, 2, utag1<
    relu<affine<con<64, 7, 7, 2, 2, SUBNET>>>>>>>>>>>>>>>>>>>>>>>;
using lr_generator_type = loss_multiclass_log_per_pixel<
    cont<256, 1, 1, 1, 1,
    generator_backbone<
    input<matrix<gray_pixel>>
    >>>;
using hr_generator_type = loss_mean_squared_per_channel_and_pixel<2,
    cont<2, 1, 1, 1, 1,
    generator_backbone<
    input<matrix<gray_pixel>>
    >>>;
// Testing network type (replaced batch normalization with fixed affine transforms)
using display_lr_generator_type = loss_multiclass_log_per_pixel<
    cont<256, 1, 1, 1, 1,
    display_backbone<
    input<matrix<gray_pixel>>
    >>>;
using display_hr_generator_type = loss_mean_squared_per_channel_and_pixel<2,
    cont<2, 1, 1, 1, 1,
    display_backbone<
    input<matrix<gray_pixel>>
    >>>;

// ----------------------------------------------------------------------------------------
// Value quantization
#define quantize(value) (static_cast<float>(value) / 127.5f - 1.0f)
#define dequantize(value) (static_cast<uint8_t>((static_cast<float>(value) + 1.0f) * 127.5f))

// ----------------------------------------------------------------------------------------
// RGB to grayscale image conversion
void rgb_image_to_grayscale_image(const matrix<dlib::rgb_pixel>& rgb_image, matrix<gray_pixel>& gray_image) {
    gray_image.set_size(rgb_image.nr(), rgb_image.nc());
    std::transform(rgb_image.begin(), rgb_image.end(), gray_image.begin(),
        [](rgb_pixel a) {return gray_pixel(a.red * 0.299f + a.green * 0.587f + a.blue * 0.114f); });
}

// RGB image <=> RGB565 image
void rgb_image_to_rgb565_image(const matrix<rgb_pixel>& rgb_image, matrix<rgb565_pixel>& rgb565_image) {
    rgb565_image.set_size(rgb_image.nr(), rgb_image.nc());
    std::transform(rgb_image.begin(), rgb_image.end(), rgb565_image.begin(), [](const rgb_pixel& p) {
        return (static_cast<uint16_t>((p.red >> 3) << 11) | static_cast<uint16_t>((p.green >> 2) << 5) | static_cast<uint16_t>(p.blue >> 3));
    });
}
void rgb565_image_to_rgb_image(const matrix<rgb565_pixel>& rgb565_image, matrix<rgb_pixel>& rgb_image) {
    rgb_image.set_size(rgb565_image.nr(), rgb565_image.nc());
    std::transform(rgb565_image.begin(), rgb565_image.end(), rgb_image.begin(), [](const uint16_t& p) {
        uint8_t red = static_cast<uint8_t>(((p >> 11) & 0x1F) << 3);
        uint8_t green = static_cast<uint8_t>(((p >> 5) & 0x3F) << 2);
        uint8_t blue = static_cast<uint8_t>((p & 0x1F) << 3);
        return rgb_pixel(red, green, blue);
    });
}
void reduce_colors(matrix<rgb_pixel>& rgb_image) {
    matrix<rgb565_pixel> rgb565_image;
    rgb_image_to_rgb565_image(rgb_image, rgb565_image);
    rgb565_image_to_rgb_image(rgb565_image, rgb_image);
}

// Function to quantize a value to 4 bits (0-15)
inline uint16_t quantize_4bits(double value) {
    return static_cast<uint16_t>(std::round(value * 15.0 / 255.0));
}
matrix<uint16_t> quantize_ab_channels(const matrix<rgb_pixel>& rgb_image) {
    matrix<lab_pixel> lab_image;
    assign_image(lab_image, rgb_image);
    matrix<uint16_t> quantized_ab_image(lab_image.nr(), lab_image.nc());
    for (long r = 0; r < lab_image.nr(); ++r) {
        for (long c = 0; c < lab_image.nc(); ++c) {
            // Pack quantized a and b channels into a single 16-bit value
            quantized_ab_image(r, c) = (quantize_4bits(lab_image(r, c).a) << 4) | quantize_4bits(lab_image(r, c).b);
        }
    }
    return quantized_ab_image;
}

#endif // DNN_MODEL_H