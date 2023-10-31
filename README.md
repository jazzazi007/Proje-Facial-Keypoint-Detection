# Project-Facial-Keypoint-Detection
Keypoint Detection is an essential project that will enable you to understand CNN and the Net architecture to apply your model in order to detect the edges in the face and set landmarks. 


nn.MSELoss() loss computes the average squared differences between the estimated values and the true values. It's a standard loss for regression problems. I used the Adam Optimizer since it was applied with a logistic regression algorithm on MNIST digit recognition data. Muhammad Yaqub et al used the Adam optimizer to detect the tumor in the grayscale images. they approved that the Adam optimizer has the highest accuracy. https://www.mdpi.com/2076-3425/10/7/427

Started with 2 conv layers and the input was one and the output was 64 with a single flattening layer. then I followed the LeNet5 architecture and the loss enhanced to 0.05. The lenet5 was applied to the first-digit images. The input of cropped images was 224*224. Then I calculated the number of reduced pixels to get the dimension of the flattened input. I used the nn.Dropout() to avoid overfitting in the raining process.
