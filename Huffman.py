string = 'BCAADDDCCACACAC'

class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        

class Item:
	def __init__(self, value, weight):
		self.value = value
		self.weight = weight

def fractionalKnapsack(W, arr):

	arr.sort(key=lambda x: (x.value/x.weight), reverse=True)

	finalvalue = 0.0

	for item in arr:

		if item.weight <= W:
			W -= item.weight
			finalvalue += item.value


		else:
			finalvalue += item.value * W / item.weight
			break
	

	return finalvalue
