import numpy as np


class PointCLoud:
    def __init__(self, points, label, mask, critical_mask):
        self.points = points
        self.label = label
        self.mask = mask
        self.critical_mask = critical_mask
        self.num_point = points.shape[0]

    def calculate(self):
        cnt_critical = 0
        cnt_backdoor = 0
        cnt_backdoor_critical = 0
        for idx in range(self.num_point):
            if self.critical_mask[idx][0] == 1. and self.mask[idx][0] == 1.:
                cnt_backdoor_critical += 1
            if not self.critical_mask[idx][0] == 1. and self.mask[idx][0] == 1.:
                cnt_critical += 1
            if self.critical_mask[idx][0] == 1. and not self.mask[idx][0] == 1.:
                cnt_backdoor += 1
        return {
            "Backdoor" : cnt_backdoor / self.num_point,
            "Critical" : cnt_critical / self.num_point,
            "Backdoor & Critical" : cnt_backdoor_critical / self.num_point,
        }

