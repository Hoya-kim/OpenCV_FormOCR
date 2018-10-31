class Cell(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.width = None
        self.height = None

        self.central_x = None
        self.central_y = None

        self.text = None
        self.text_height = None
        self.text_align = 'center'
        self.text_valign = 'center'

        self.boundary = {
            'left': False,
            'right': False,
            'upper': False,
            'lower': False
        }



class Preprocessing():
    def __init__(self):
        #Cell.__init__(self)
        self.final_x = 10
        self.final_y = 10
        self.cells = None

    def rrr(self):
        self.cells = [[Cell() for cols in range(self.final_y)] for rows in
                      range(self.final_x)]
        for i in range(10):
            print(i)
            for j in range(10):
                print(self.cells[i][j])


pp = Preprocessing()
pp.rrr()
