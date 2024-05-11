class ObjectiveFunction:
    @staticmethod
    def evaluate(SA):
        output = 0.0
        for s in SA:
            output += s * s
        return output
