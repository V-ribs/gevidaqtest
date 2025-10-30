"""Make sure to always perform a new refistration as soon as the hardware has changed or moved!"""


class CameraPmtRegistrationPoints:
    def __init__(self):

        use_random_test_values = False
        registration_25_2_1st = True

        if use_random_test_values:
            self.pmt_vertices = [(133, 43), (163, 193), (76, 56)]
            self.camera_vertices = [(748, 664), (1260, 1172), (559, 869)]

        elif registration_25_2_1st:
            self.pmt_vertices = [(0, 0), (250, 500), (500, 100)]
            self.camera_vertices = [(1422, 1643), (570, 1210), (1256, 769)]
