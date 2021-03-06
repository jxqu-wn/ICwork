import random
import math
from geopy.distance import geodesic


xMax = 200
yMax = 200


class Dustbin:
    def __init__(self, x=None, y=None):
        if x is None and y is None:
            self.x = random.randint(0, xMax)
            self.y = random.randint(0, yMax)
        else:
            self.x = x
            self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceTo(self, db):
        dis = geodesic((self.getX(), self.getY()), (db.getX(), db.getY())).km
        return dis

    def toString(self):
        s = '(' + str(self.getX()) + ',' + str(self.getY()) + ')'
        return s

    def checkNull(self):
        if self.x == -1:
            return True
        else:
            return False
