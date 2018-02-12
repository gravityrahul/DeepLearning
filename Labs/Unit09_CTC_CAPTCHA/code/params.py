class params:
    
    def __init__(self):
        # Input parameters
        self.img_h = 64
        self.img_w = 512
        self.epochs = 20
        self.batch_size = 32
        self.words_per_epoch = 16000
        self.val_split = 0.2
        self.val_words = int(self.words_per_epoch * (self.val_split))
        self.train_words = self.words_per_epoch - self.val_words
        self.absolute_max_string_len = 16
        
        # Network params
        self.filters = 16
        self.kernel_size = (3, 3)
        self.pool_size_1 = (4, 4)
        self.pool_size_2 = (2, 2)
        self.time_dense_size = 32
        self.rnn_size = 512
        self.time_steps = self.img_w // (self.pool_size_1[0] * self.pool_size_2[0])
        
    def _update(self):
        self.time_steps = self.img_w // (self.pool_size_1[0] * self.pool_size_2[0])
        self.val_words = int(self.words_per_epoch * (self.val_split))
        self.train_words = self.words_per_epoch - self.val_words
    
    def __str__(self):
        def display(objects, positions):
            line = ''
            for i in range(len(objects)):
                line += str(objects[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            return line
        
        line_length = 100
        ans = '-' * line_length
        members = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        for field in members:
            if field[0] != "_":
                objects = [field, getattr(self, field)]
                positions = [30, 100]
                ans += "\n" + display(objects, positions)
        ans += "\n" + '-' * line_length
        return ans