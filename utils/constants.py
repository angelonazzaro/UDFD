MODEL_NAME: str = "dima806/deepfake_vs_real_image_detection"
RACES = {
    "white": 0,
    "black": 1,
    "latino hispanic": 2,
    "middle eastern": 3,
    "asian": 4,
    "indian": 5,
}
INV_RACES = {
    0: "white",
    1: "black",
    2: "latino hispanic",
    3: "middle eastern",
    4: "asian",
    5: "indian",
}

GENDERS = {
    "man": 0,
    "woman": 1,
}
PATIENCE = 5
IMG_SIZE = 224
