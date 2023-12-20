
class Status:
    
    def __init__(self):    
        self.downloading = False
        self.processing = False
        self.downloadProgress = 0.0
        self.processProgress = 0.0
    
    def update(self, downloading, processing, downloadProgress, processProgress):
        self.downloading = downloading
        self.processing = processing
        self.downloadProgress = downloadProgress
        self.processProgress = processProgress
        
    def get_status(self):
        dict = {'downloading': self.downloading, 
                'processing': self.processing,
                'downloadProgress': self.downloadProgress,
                'processProgress': self.processProgress}
        print(id(self), dict)
        return dict
    
status = Status()