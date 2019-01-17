class Rectangle:
    count = 0  # 클래스 변수

    def __init__(self, width, height):
        self.width = width
        self.height = height
        Rectangle.count += 1

        print('yes!!!!')

    # 인스턴스 메서드
    def calcArea(self):
        area = self.width * self.height
        return area


    # 정적 메서드
    @staticmethod
    def isSquare(rectWidth, rectHeight):
        return rectWidth == rectHeight

        # 클래스 메서드

    @classmethod
    def printCount(cls):
        print(cls.count)
