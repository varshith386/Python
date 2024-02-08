Folders:
	Input:
		Train: Dataset 1
		Data:  Dataset 2

	Output Folders:
			Out5,Out6,Out5_a,Out6_a,Out7,Out7_a


Change the paths before running the code

3 Methods (StackReq,CV2,FFT) Applied on 2 datasets (i.e Dataset 1, Dataset2)

Evaluation Metrices: Overlap Measure | Jaccard Index[A.K.A --IOU(Intersection Over Union)]
	             Euclidian Measure | MSE(Mean Square Error)
		     SAD(Sum of Absolute Difference) | SSIM(Structural Similarity Index)


#NOTE
        #Certain evaluation Metrices arent compatible with certain Models
	 Hence, few are avoided and alternate are used
	 Four Evaluation are used for each Models
	
	#Use smaller dataset for faster results