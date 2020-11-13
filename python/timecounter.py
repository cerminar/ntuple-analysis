import time


class TimeCounter(object):
    def __init__(self):
        self.start_tic = 0
        self.start_real = 0
        self.job_flavors = {
            'test': 6*60-30,        # 1 minutes
            'espresso': 20*60,        # 20 minutes
            'microcentury': 1*60*60,  # 1 hour
            'longlunch': 2*60*60,     # 2 hour
            'workday': 8*60*60,       # 8 hour
            'tomorrow': 24*60*60,     # 1 days
            'testmatch': 3*24*60*60,  # 3 days
            'nextweek': 7*24*60*60,   # 1 week
            }
        self.job_flavors_labels = {
            'test': '(1 minutes)',        # 1 minutes
            'espresso': '(20 minutes)',        # 20 minutes
            'microcentury': '(1 hour)',  # 1 hour
            'longlunch': '(2 hours)',     # 2 hour
            'workday': '(8 hours)',       # 8 hour
            'tomorrow': '(1 days)',     # 1 days
            'testmatch': '(3 days)',  # 3 days
            'nextweek': '(1 week)',   # 1 week
            }

        pass

    def start(self):
        self.start_tic = time.perf_counter()
        self.start_real = time.time()

    def real_time(self):
        return time.time() - self.start_real

    def time(self):
        return time.perf_counter() - self.start_tic

    def time_per_event(self, nevents):
        analysis_time = self.time()
        return analysis_time, analysis_time/nevents

    def job_flavor_time(self, flavor):
        return self.job_flavors[flavor]

    def print_nevent_per_jobflavor(self, time_per_event):
        if self.start_tic != 0:
            for job_flavor, job_time in self.job_flavors.items():
                print ("{} {}: # ev: {}".format(job_flavor, self.job_flavors_labels[job_flavor], int(job_time/(1.1*time_per_event))))

    def job_flavor_time_perc(self, flavor):
        if self.start_tic != 0:
            time = self.time()
            flavor_time = self.job_flavor_time(flavor)
            print ('Job time: {:.2f}: {:.1f}% of flavor {}'.format(time, 100*time/flavor_time, flavor))
            return time/flavor_time

    def job_flavor_time_left(self, flavor):
        if self.start_tic != 0:
            time = self.time()
            flavor_time = self.job_flavor_time(flavor)
            return flavor_time - time

    def started(self):
        return self.start_tic != 0


counter = TimeCounter()
