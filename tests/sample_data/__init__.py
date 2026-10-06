import os
import glob

_de_cat = os.path.dirname(__file__)
data_archives = glob.glob(os.path.join(_de_cat, "*.tar.bz2"))
cepa_archives = glob.glob(os.path.join(_de_cat, "cepa*.tar.bz2"))