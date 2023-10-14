from locust import HttpUser, task, between
import cv2

def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))

    # 去除元信息（strip metadata）
    # 如果图像包含元信息，可以使用Exif标记来删除它
    # 如果没有元信息，可以跳过这一步
    # 示例代码：
    # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    return resized_image

class QuickstartUser(HttpUser):
    #wait_time = between(1, 2)


    def on_start(self):
        image = cv2.imread("test_human_2.jpeg")
        new_width = 1024
        resized_img = resize_img(image, new_width)

        # lower the quality
        quality = 65  # quality set to 65%
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        _, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)
        self.img_bytes = jpeg_frame.tobytes()


    @task
    def hello_world(self):
        processing_endpoint = 'http://localhost:8848/process_video'
        self.client.post("/process_video", data=self.img_bytes,
                                 headers={'Content-Type': 'image/jpeg'})

    # @task(3)
    # def view_item(self):
    #     for item_id in range(10):
    #         self.client.get(f"/item?id={item_id}", name="/item")