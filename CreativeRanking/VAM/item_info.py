from collections import defaultdict

class creative_info(object):
    """creative_info
    creative info includes pv and click.
    """
    def __init__(self):
        self.pv = 0
        self.clk = 0
    def update(self, pv, clk):
        self.pv += pv
        self.clk += clk

class item_info(object):
    """item_info
    item info includes candidate images.
    """
    def __init__(self):
        self.creatives = defaultdict(creative_info)
        self.creative_num = 0
        self.item_pv = 0

    def add_creative(self, img, pv, clk):
        if img not in self.creatives:
            self.creative_num += 1
        self.creatives[img].update(pv, clk)
        self.item_pv += pv

    def filter_creative(self, pv_thresh):
        self.creatives = {k: v for k, v in self.creatives.items() if v.pv >= pv_thresh}
        self.creative_num = len(self.creatives.keys())
        self.item_pv = sum([self.creatives[k].pv for k in self.creatives.keys()])


