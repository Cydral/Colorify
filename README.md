# Colorify - AI-Powered Image Colorization with Dlib
![image](https://github.com/Cydral/FFspider/assets/53169060/532c096d-d06f-433c-902a-049985cd26c7)
<p><i>Colorify is a black and white image colorizer based on Dlib. It uses a generative adversarial network (GAN) to learn the mapping from black and white images to color images. The GAN is trained on a dataset of over 250,000 images, and it is able to colorize images in a realistic and natural way.</i></p>

<h2>Description</h2>
<p>Colorify is a C++ program that leverages the power of generative adversarial networks (GANs) to automatically colorize black and white images using a DCGAN (Deep Convolutional Generative Adversarial Network) architecture. The program takes a grayscale image as input and generates a corresponding colorized version where the pixel values have been determined by a generative AI model.</p>

<h2>Overview</h2>
<p>The Colorify program is designed to infuse life into monochromatic images by applying state-of-the-art image colorization techniques. It employs a DCGAN, a type of GAN composed of two networks: a Generator and a Discriminator. The Generator network is built on a U-Net architecture, while the Discriminator network utilizes a series of filters with or without activation. This combination allows Colorify to learn and replicate intricate color patterns, producing visually appealing colorizations.</p>
<p align="center"><img src="https://github.com/Cydral/Colorify/blob/main/artictic-depiction-GAN-process.jpg" alt="Artistic depection of GAN process for color image generation"></p>

<h2>Training Data and Process</h2>
<p></p>The training data for the Colorify model consists of a diverse set of +250k images collected from the Web. These images were randomly extracted using the FFspider program, available <a href="https://github.com/Cydral/FFspider">here</a>. This rich dataset serves as the foundation for teaching the AI model to understand color relationships and patterns, enabling it to generate accurate and vibrant colorizations.</p>
<p>To enhance the diversity and robustness of the training data, an augmentation method was employed. This method involves extracting random sub-zones of the images, ranging from 70% to 100% of the original size. This augmentation strategy enriches the dataset, enabling the AI model to grasp color relationships and patterns more comprehensively</p>
<p>During the training process, Colorify adopts a unique approach to channel decomposition by utilizing the CIE L*a*b space. Similarly to traditional methods, the Generator network receives the luminance channel directly as input and learns to generate the a and b channels as output. The Discriminator network, on the other hand, is trained to differentiate between real images and generated images based solely on the a and b channels. This novel approach was informed by our own evaluations, which demonstrated that training the discriminator solely on the L channel had little impact on the learning process. Moreover, this approach simplifies the overall complexity of the training process without compromising convergence capability.</p>

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
