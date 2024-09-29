import cv2
import numpy as np

# Load the original image
for relation in 'Comparison Contingency Expansion Temporal'.split():
    image_path = f'/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/baseFirst_dev_None_withoutS/{relation}.dev_score.png'  # Replace with your image path
    image_path = f'/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/baseFirst_train_None_draw2/{relation}.train_score.png'
    # image_path = image_path.replace('xxxxx', relation)
    # print(image_path)
    # continue
    # image_path = r'D:\0--data\projects\paper\Subtext\figs\baseFirst_train_None_draw2\Comparison.train_score.png'
    image = cv2.imread(image_path)

    # Get the dimensions of the original image
    height, width, _ = image.shape
    print(image.shape)
    # continue
    # exit()

    # left_margin = 14  # Increase left margin
    # top_margin = 0    # No change in top margin
    # right_margin = 14  # No change in right margin
    # bottom_margin = 0  # No change in bottom margin
    # Create a new image with increased size
    # new_height = height + top_margin + bottom_margin
    # new_width = width + left_margin + right_margin

    new_height = 553
    new_width = 918
    assert new_height >= height and new_width>=width
    left_margin = right_margin = (new_width-width)//2 
    top_margin = bottom_margin = (new_height-height)//2   
    
    new_image = np.ones((new_height, new_width, 3), dtype=np.uint8)*255

    # Place the original image in the new image with the increased left margin
    new_image[top_margin:top_margin + height, left_margin:left_margin + width] = image

    # Save the modified image

    # Optionally, display the modified image
    print(new_image.shape)
    # cv2.imshow('Modified Image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(image_path, new_image)  # Replace with your output path