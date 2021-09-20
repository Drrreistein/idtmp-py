
from z3 import *
from IPython import embed
def main():
    x, y = Ints('x y')
    s = Solver()
    s.add(x+y==10, x<10, y<10)
    vars = [x, y]
    while sat==s.check():
        m = s.model()
        print(m, '\n')
        res = []
        for var in vars:
            res.append(var == m[var])
        s.add(Not(And(res)))

if __name__=='__main__':
    main()
