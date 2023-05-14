
        self.pil_imgs = [Image.fromarray(cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB)) for image in self.images]
        self.photos = [ImageTk.PhotoImage(pil_img) for pil_img in self.pil_imgs]
