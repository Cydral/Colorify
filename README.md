# Colorify - AI-Powered Image Colorization with Dlib
![image](https://github.com/Cydral/FFspider/assets/53169060/532c096d-d06f-433c-902a-049985cd26c7)
<p><i>Colorify is a versatile image colorization program built upon Dlib. It offers a spectrum of advanced techniques for image colorization, including supervised learning methods like classification and regression, alongside cutting-edge self-supervised generative adversarial networks (GANs). Our program empowers users to create two distinct models: one for efficient low-resolution colorization (up to 65K colors) and another for high-resolution colorization, capable of producing images with rich detail and a palette of over 262K colors.
Our approach leverages a robust dataset comprising more than 250,000 images, ensuring that our models can colorize images with striking realism and naturalness. Whether you're looking to add subtle tints or bring life to monochrome photographs, Colorify offers a comprehensive solution for all your image colorization needs. Dive into the world of vibrant colors and unleash your creativity with Colorify.</i></p>

<h2>Description</h2>
<p>Colorify is a comprehensive C++ application that harnesses the capabilities of generative models to perform automated colorization of black and white images. It employs a sophisticated neuronal architecture to achieve this task. When given a grayscale image as input, Colorify utilizes generative AI models (two pre-calculated models are provided that can also be fine-tuned) to produce a fully colorized version.</p>
<p>The program opens the way to a multitude of use cases. Although it excels in static image enhancement, its potential extends far beyond this, as it can also be used for automated movie colorization. Drawing on the latest developments in the Dlib image processing library, we have also provided a dedicated "ready-to-use" program, designed specifically for video colorization. This comprehensive suite enables users to bring individual images and entire film sequences to life with unrivalled ease and efficiency.</p>

<h2>Overview</h2>
<p>The Colorify program takes a multi-faceted approach to revitalizing monochromatic images, by employing three distinct methods to generate different models. For the "low resolution" model, we employ a supervised learning approach widely recognized in the literature for this purpose. However, instead of traditional color quantization based on the determination of an index, we implement a color reduction technique. This method automatically creates an index by compressing the "a" and "b" channel values of the Lab color model. The approach simplifies the calculation, which is similar to that used in segmentation, but limits the color palette to a maximum of 65,000 different colors, hence the name "low resolution" for this model.</p>
<p>For those looking for images with a wider, more vibrant color spectrum, we propose an alternative approach. The other method combines supervised learning, in particular regression, to initially calculate a basic model. We then refine this model using a GAN, resulting in a "high-resolution" model capable of producing images with a rich and detailed color range, potentially up to 16 million colors.</p>
<p>Whichever method you choose, the program also offers a progressive display of the results and the performance of the current learning model for the colorization task. Additionally, the backbone used for understanding the intrinsic features of the image can be extracted and injected into each learning process (transfer learning), allowing you to leverage the outcomes of previous training for enhanced performance.</p>
<p align="center"><img src="https://github.com/Cydral/Colorify/blob/main/highres-model-training-1.jpg" alt="Color image generation during the training process"></p>

<h2>Training Data and Process</h2>
<p>The training data for the Colorify model consists of a diverse set of +250k images collected from the Web. These images were randomly extracted using the FFspider program, available <a href="https://github.com/Cydral/FFspider">here</a>. This rich dataset serves as the foundation for teaching the AI model to understand color relationships and patterns, enabling it to generate accurate and vibrant colorizations.</p>
<p>To enhance the diversity and robustness of the training data, an augmentation method was employed. This method involves extracting random sub-zones of the images, ranging from 85% to 100% of the original size. This augmentation strategy enriches the dataset, enabling the AI model to grasp color relationships and patterns more comprehensively</p>
<p>During the training process, Colorify adopts a unique approach to channel decomposition by utilizing the CIE L*a*b space. Similarly to traditional methods, the Generator network receives the luminance channel directly as input and learns to generate the a and b channels as output. The Discriminator network, on the other hand, is trained to differentiate between real images and generated images based solely on the a and b channels. This novel approach was informed by our own evaluations, which demonstrated that training the discriminator solely on the L channel had little impact on the learning process. Moreover, this approach simplifies the overall complexity of the training process without compromising convergence capability.</p>

<h2>Disclaimer</h2>
<p>Please note that Colorify's colorization algorithm is primarily trained on a dataset consisting of images randomly selected from web sites. While it performs well for similar types of images, it may not yield optimal results for all image types due to the inherent "dataset bias" problem, as mentioned in the document <a href="https://arxiv.org/abs/1603.08511">Colorful Image Colorization</a> paper from Richard Zhang, Phillip Isola and Alexei A. Efros.</p>
<p>The algorithm may also produce impressive colorizations; there might occasionally occur what we term as "chromatic aberration." This phenomenon can manifest as traces of blurriness in certain areas of the image along with color saturation. This is mainly attributed to localized "neural saturation" failing to recognize specific patterns. We include these aspects as part of our experimental observations and findings in the colorization process.</p>
<p>Furthermore, despite the substantial size of the training dataset used for the provided DNN model, a "gray-toned effect" (or brownish) might also manifest. This effect reflects a statistical uncertainty regarding the color that the neural network should generate for the specific area (or even the entire image) being processed.</p>

<h2>Contributions</h2>
Contributions to Colorify are welcome! If you'd like to enhance the program or fix any issues, please follow these steps:
<ul>
  <li>Fork the repository.</li>
  <li>Create a new branch for your feature or bug fix.</li>
  <li>Make your changes and commit them.</li>
  <li>Push your changes to your forked repository.</li>
  <li>Create a pull request, detailing the changes you've made.</li>
</ul>
Enjoy with the Colorify framework!

<h2>License</h2>
<p>Colorify is released under the <a href="https://github.com/Cydral/Colorify/blob/main/LICENSE">MIT License</a>. See the LICENSE file for more details.</p>
