# from multiprocessing import Pool
# POOL = Pool(5)

# FIXME: dummy to avoid the complication of multiprocessing for now
class Pool(object):
    def map(self, func, args):
        ret = []
        for arg in args:
            ret.append(func(args))
        return ret


POOL = Pool()
