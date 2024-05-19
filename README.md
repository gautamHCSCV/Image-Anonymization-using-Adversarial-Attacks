[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/AQPBb0Hq)

## Image Anonymization using Adversarial Attacks

In the digital age we live in now, more and more image data has brought both rewards and problems. Even though images have useful information for analyzing and understanding, they can also have private or personal information that needs to be kept safe. Image anonymization methods are a key part of protecting people's privacy and making sure that data protection laws are followed. This project's goal is to look into the field of picture anonymization and come up with a new way to do it using the Projected Gradient Descent (PGD) attack. By using methods for adversarial robustness, we can change images in a way that keeps their visual content but makes it hard to tell what they are. 

Image anonymization is important for protecting privacy in today's world where images are widely shared and analyzed. It involves removing or changing personal details like faces or license plates so that individuals' identities are safeguarded, and their information is not misused or accessed without permission. It helps organizations comply with privacy regulations like the GDPR, ensuring sensitive data isn't exposed. Anonymizing images is also crucial for ethical AI, as it prevents biases, unintended disclosure of sensitive information, and helps build fair and unbiased machine learning models that use image data.

Image anonymization has important applications in various areas. In healthcare, it is used to protect patients' identities and medical records when sharing medical images for research or collaboration. In public surveillance, anonymization techniques can blur faces captured by video surveillance systems, ensuring individuals' identities are safeguarded while still allowing analysis for security purposes. Social media platforms use image anonymization to protect user privacy, preventing unauthorized identification or misuse of personal data. Additionally, researchers benefit from anonymizing images before sharing datasets, as it ensures privacy while supporting advancements in fields like computer vision and machine learning.


### Adversarial Attacks
In mobile and pervasive computing, adversarial attacks are when bad actors purposely try to harm or exploit weaknesses in mobile devices and their systems. They do this by creating malicious software, taking advantage of vulnerabilities in networks, tricking people into giving away information, or tampering with mobile applications. These attacks can have serious consequences, like stealing personal information, damaging the device's functionality, or disrupting network communication. For example, attackers might create fake apps that look real but are actually designed to steal sensitive data. They might also intercept data being sent between devices and networks to gain unauthorized access or manipulate the information.


### PGD Attack
The PGD (Projected Gradient Descent) attack is a powerful iterative technique used in adversarial machine learning to generate adversarial examples. It is an extension of the Fast Gradient Sign Method (FGSM) and aims to create more robust and effective adversarial perturbations. In the PGD attack, instead of making a one-time perturbation to the input, multiple iterations are performed to refine the perturbation. The process involves taking small steps in the direction of the gradient of the loss function with respect to the input, while constraining the perturbation to stay within a predefined range or budget. At each iteration, the perturbation is projected back onto the allowed range to ensure it remains within the specified boundaries.

The PGD attack is more powerful than the FGSM because it explores a larger space of possible perturbations. By performing multiple iterations and projecting the perturbation back onto the feasible range, the attacker can find adversarial examples that are more likely to fool the targeted machine learning model while still being within the defined limits.

### Results
We trained 2 state of the art deep learning models (Squeezenet and ShuffleNet) on CIFAR-10 dataset and saw the effect of various attacks on these datasets with different epsilon values.
![image](https://github.com/Mobile-and-Pervasive-Computing-Projects/course-projects-gautamHCSCV/assets/65457437/d754d166-91fa-442e-b0e7-b8dda1e24eec)

From the results obtained, we can observe that PGD attack is more powerful than
the mask based attack. PGD attack uses a more sophisticated approach to generate
adversarial examples. It uses an iterative process to find the optimal perturbation
that can fool a deep learning model, while the mask based attack only applies a
single perturbation. PGD also allows for more flexibility in terms of the magnitude of
the perturbation, which makes it more effective in fooling deep learning models.

We also performed the perturbation on YOLO-v7 detections using the PGD attack and observed the changes. We were successfully able to misclassify the given entities in a given image. The corresponding confidence score also goes down drastically.

### Steps to Run the codes
The code will run on a regular google colab or jupyter notebook. All the python files are independent of each other.

Dependencies:

NumPy: 1.19.5 </br>
Pandas: 1.1.5 </br>
Matplotlib: 3.2.2 </br>
PyTorch: 1.9.0 </br>
Scikit-learn: 0.24.2 </br>
OpenCV: 4.5.3 </br>

Installation for the flutter: </br>
Getting Started </br> </br>

To get started with the app, follow these steps: </br> </br>

    Clone the repository: git clone [https://github.com/your-username/flutter-app.git](https://github.com/Mobile-and-Pervasive-Computing-Projects/course-projects-gautamHCSCV.git) </br>
    Navigate to the project directory: cd flutter-app </br>
    Install the dependencies: flutter pub get </br>
    Run the app: flutter run </br>
 </br>
To run the object detector model use the following steps before using the given colab file:

Mounting Google Drive

from google.colab import drive
drive.mount('/content/gdrive')

%%bash

cd /content/gdrive/MyDrive </br>
git clone https://github.com/WongKinYiu/yolov7.git </br>
cd yolov7 </br>
wget https://raw.githubusercontent.com/WongKinYiu/yolov7/u5/requirements.txt </br>
pip install -r requirements.txt </br>

import os </br>
import sys </br>
sys.path.append('/content/gdrive/MyDrive/yolov7') </br>

%%bash </br>
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt </br>

wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt </br>

wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt </br>


![image](https://github.com/Mobile-and-Pervasive-Computing-Projects/course-projects-gautamHCSCV/assets/65457437/8b08d918-8d77-4cf9-ad90-8c8e97bdae52)

![image](https://github.com/Mobile-and-Pervasive-Computing-Projects/course-projects-gautamHCSCV/assets/65457437/c7640884-ee73-453b-8138-9b9f20f85e8b)



### CONCLUSION
Our image anonymization project utilizing the PGD attack has yielded successful results in generating perturbations and introducing changes to images. We observed that as we increased the epsilon value, which determines the magnitude of the perturbation, the test accuracy of the machine learning model decreased. This indicates that the generated perturbations were effective in altering the model's predictions and potentially anonymizing the images.  The decrease in test accuracy demonstrates the vulnerability of the model to adversarial attacks and highlights the importance of considering robust defenses against such attacks in image anonymization techniques. It suggests that the perturbations introduced through the PGD attack can effectively deceive the model and potentially protect the privacy and identities of individuals in the images.

Our project contributes to the field of image anonymization by demonstrating the potential of the PGD attack and its impact on test accuracy. The findings emphasize the need for robust defenses and careful considerations when implementing image anonymization techniques to ensure both privacy protection and reliable model performance.



