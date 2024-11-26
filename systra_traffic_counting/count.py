from collections import defaultdict
from tqdm import tqdm
class Counter:
    """
    A class to count objects crossing a specified tripline.
    Attributes:
    -----------
    START : tuple
        The starting point of the tripline.
    END : tuple
        The ending point of the tripline.
    CROSSED : dict
        A dictionary to store the track IDs of objects that have crossed the tripline.
    Methods:
    --------
    __init__(tripline):
        Initializes the Counter with the tripline position.
    intersect_tripline(A, B):
        Determines if the tripline intersects with a segment defined by points A and B.
    count(track_data, progress_bar=None):
        Counts the number of objects that cross the tripline based on the provided track data.
    """
    def __init__(self, data_manager):
        self.START, self.END = data_manager.START, data_manager.END
        self.directions = data_manager.directions

    # Return true if tripline and segment AB intersect
    def intersect_tripline(self, A, B):
        def ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(self.START,A,B) != ccw(self.END,A,B) and ccw(self.START,self.END,A) !=  ccw(self.START,self.END,B)

    def count(self, data_manager, progress_var = None):
        self.progress = progress_var
        self.console_progress = tqdm(total=len(data_manager.TRACK_DATA), desc="Checking each object track for crossing", unit="tracks")
        self.length, prog_counter = len(data_manager.TRACK_DATA), 0
        for track_id, data in data_manager.TRACK_DATA.items() :
            if self.intersect_tripline(data[0][1],data[-1][1]) : #Check that they actually cross tripline, otherwise skip them
                finish = (data[-1][1][0],data[-1][1][1])
                i = len(data)-1
                while not self.intersect_tripline((data[i-1][1][0],data[i-1][1][1]),finish): # Find the first frame where they crossed
                    i -= 1
                direction = self.directions[0] if data[i][1][1] < data[i-1][1][1] else self.directions[1]
                data_manager.CROSSED[track_id] = (data[i][0], data[i][3], direction) #(frame_nb, class, direction)
            self.console_progress.update(1)
            prog_counter += 1
            if self.progress is not None:
                    self.progress.set(prog_counter)
        self.console_progress.close()


    