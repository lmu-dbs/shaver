class DB:
    def __init__(self, log, mapping_dict, traces_with_timestamps, avg, med, std, ext, mysum, mymax, title) -> None:
        self.avg = avg
        self.med = med
        self.std = std
        self.ext = ext
        self.mysum = mysum
        self.mymax = mymax
        self.title = title
        self.traces_with_timestamps = traces_with_timestamps
        self.log = log
        self.mapping_dict = mapping_dict


