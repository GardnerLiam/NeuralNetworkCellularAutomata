import numpy as np
from tqdm import tqdm
from cell import Cell
import matplotlib.pyplot as plt

def draw(grid,iteration=0):
    plt.imshow(grid)
    plt.axis('off')
    plt.title("Iteration {}.png".format(iteration))
    plt.savefig('Figures/fig{:04d}.png'.format(iteration))

if __name__ == "__main__":
    mx = 10
    my = 10

    gc = 100
    ncells = 5

    grid = np.ones((mx+1,my+1,3)).astype(np.uint32)*gc

    cells = []
    for i in range(ncells):
        x = np.random.randint(0,mx)
        y = np.random.randint(0,my)
        s = np.round(np.random.random(),2)
        cells.append(Cell(x,y,s,mx,my))
        grid[x][y] = [int(127+s*127),0,0]

    draw(grid)
    for i in tqdm(range(100)):
        subcells = []
        for c in cells:
            if (not c.isDead):
                u = [c.x,my]+list(grid[c.x,my]) if c.y == 0 else [c.x,c.y-1]+list(grid[c.x,c.y-1])
                r = [0,c.y]+list(grid[0,c.y]) if c.x == mx else [c.x+1,c.y]+list(grid[c.x+1,c.y])
                d = [c.x,0]+list(grid[c.x,0]) if c.y == my else [c.x+1,c.y]+list(grid[c.x,c.y])
                l = [mx,c.y]+list(grid[mx,c.y]) if c.x == 0 else [c.x-1,c.y]+list(grid[c.x-1,c.y])
            
                grid[c.x,c.y] = [gc,gc,gc]

                px = c.x
                py = c.y
                
                c.move([u,r,d,l])
                grid[c.x,c.y] = [int(127+c.s*127),0,0]
                if (c.life == 3):
                    subcells.append(c.birth(px,py))

                for c_2 in cells:
                    if c.x == c_2.x and c.y == c_2.y and c.s != c_2.s:
                        if (c.s > c_2.s):
                            c.NN.average(c_2.NN.layers)
                            c_2.isDead = True
                            c_2.x = -1
                            c_2.y = -1
        draw(grid,i+1)
        for i in range(len(cells)-1, -1, -1):
            if (cells[i].isDead):
                cells.pop(i)
        cells += subcells


