TOPOLOGY_NAME = "gaia"
EXPERIMENT = "femnist"
LOCAL_STEPS = 1
T_MAX = 5
ARCH = "ring"

# Model size in bit
MODEL_SIZE_DICT = {
        "femnist": 4843243,
        "sent140": 19269416,
        "inaturalist": 44961717,
        "shakespeare": 3385747,
        "driving_carla": 10091936,
        "synthetic": 4354
    }

# Model computation time in ms
COMPUTATION_TIME_DICT = {
        "femnist": 4.6,
        "sent140": 9.8,
        "inaturalist": 25.4,
        "shakespeare": 389.6,
        "driving_carla": 7.2,
        "synthetic": 1.5
    }

ROUNDS_DICT = {
        "femnist": 6400,
        "sent140": 20000,
        "inaturalist": 1600,
        "shakespeare": 1500,
        "driving_carla": 1600,
        "synthetic": 3000
    }