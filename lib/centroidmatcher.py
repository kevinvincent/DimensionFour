# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidMatcher():

	def match(self, objects, inputs):

		newInputs = []

		# initialize an array of input centroids for the current frame
		objectCentroids = np.zeros((len(objects), 2), dtype="int")
		inputCentroids = np.zeros((len(inputs), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(objects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			objectCentroids[i] = (cX, cY)

		for (i, (startX, startY, endX, endY)) in enumerate(inputs):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)


		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(objects) == 0:
			for input in inputs:
				newInputs.append(input)

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			
			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# Find new inputs
			if D.shape[0] < D.shape[1]:
				for col in unusedCols:
					for (i, (startX, startY, endX, endY)) in enumerate(inputs):
						# use the bounding box coordinates to derive the centroid
						cX = int((startX + endX) / 2.0)
						cY = int((startY + endY) / 2.0)
						if inputCentroids[col] == (cX, cY):
							newInputs.append((startX, startY, endX, endY))

		# return the set of trackable objects
		return newInputs