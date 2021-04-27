
class BoostingMetric(object):

    def __init__(self):
        self.profile = profile

    @profile
    def training_time(self, *args):
        return args

