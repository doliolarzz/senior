class DataGenerator():

    def __init__(self, data_path, k_fold, batch_size, in_len, out_len):

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.in_len = in_len
        self.out_len = out_len
        self.current_k = 1

    def set_k(self, k):
        self.current_k = k

    def get(self, i):
        #Clean previous by del
        #get new one and return
        return None, None

    def shuffle(self):
        None
        
    def size(self):
        None