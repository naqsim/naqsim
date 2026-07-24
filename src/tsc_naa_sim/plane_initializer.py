from tsc_instructions import *

def get_rectangle(n, max_diff):
    min_area = float('inf')
    best_dims = None
    for h in range(1, n + 1):
        min_w = h
        max_w = h + max_diff

        for w in range(min_w, max_w + 1):
            area = w * h
            if area >= n and area < min_area:
                min_area = area
                best_dims = (w, h)

            if area >= n:
                break

    return best_dims

class plane_initializer:
    def __init__(self, 
                 num_lq, 
                 plane_type
                 ):
        # input
        self.num_lq = num_lq
        self.plane_type = plane_type
        # output
        self.plane_char = None
    
    def generate_plane_char(self):
        # shape
        ## row1: m port, row2: y port
        ## sparsity - Assume DENSE (no empty space)
        ## aspect ratio - Assume square
        num_init_qregs = self.num_lq
        if self.plane_type == PlaneType.ALLQ:
            width, height = get_rectangle(num_init_qregs, max_diff = 2)
        else:
            raise Exception()
        height += 2    
        ##
        # generate
        self.plane_char = [['.' for _ in range(width)] for _ in range(height)]
        for r in range(height):
            for c in range(width):
                if r == 0: # m ports
                    self.plane_char[r][c] = 'M'
                elif r == 1: # y ports
                    self.plane_char[r][c] = 'Y'
                else: # normal 
                    self.plane_char[r][c] = 'N'
        #
        return
    
    def run(self):
        self.generate_plane_char()