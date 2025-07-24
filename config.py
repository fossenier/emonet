"""
Config for the iSE usage of emonet.

Logan Fossenier July 2025
"""


class Config:
    def __init__(self) -> None:
        self.pretrained_path: str = (
            "/u2/users/hzv143/parnian/pretrained/emonet_8.pth"  # state dict
        )
        self.emotion_classes = {
            0: "Neutral",
            1: "Happy",
            2: "Sad",
            3: "Surprise",
            4: "Fear",
            5: "Disgust",
            6: "Anger",
            7: "Contempt",
        }  # hard coded to the 8 classification mode
        self.image_size = 256  # 256 x 256 input
        self.device = "cuda:2"  # use the A100
        self.emotion_count = len(self.emotion_classes)  # which classification mode
