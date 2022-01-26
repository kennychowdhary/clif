import numpy as np
from sklearn.decomposition import PCA

class fingerprints:
	''' Class for fingerprinting and computing Empircal Orthogonal Functions (EOFs) from data
	
	Given a 2d data set we perform PCA and return the empirical orthogonal functions. The data set should be in form (n_samples, latitude x longitude), so that the second dimension is a combination of all latitude and longitude points. 

	Parameters
	----------
	n_eofs	: 	str, default = 2

		Number of empirical orthogonal functions to return, i.e. number of principal components

	whiten	: 	bool, default = False

		Whether to normalize the projection coefficients of the EOFs. 

	varimax	:	bool, default = False

		Perform Varimax rotation after PCA is perform to obtain sparse entries


	Notes
	-----
	For now, the PCA will use an automatic solver, with default parameters in sklearn's PCA. In general, we allow the user to specify a different method.

	To do
	-----
	* add option for using xarray as input
	* add option for multiple outputs
	* add option for factor analysis and other methods
	* implement varimax procedure into code using the factor analysis toolbox in sklearn

	
	'''
	def __init__(self,n_eofs=2,varimax=False,method='pca',method_opts={'whiten':True,'svd_solver':'arpack'}):
		self.n_eofs = n_eofs
		self.method_opts = method_opts
		self.varimax = varimax
		self.method = method
	def fit(self,X):
		''' Perform EOF decomposition for a numpy array

		To do
		-----
		* Allow for xarray data set (n,lat,lon)
		* allow for list of data X = [X1,X2,...] for multivariate fingerprinting
		'''
		# perform a check on the data
		assert X.ndim == 2, "Input data matrix is not 2 dimensional."
		self.n_samples, self.n_dim = X.shape
		if self.method == 'pca':
			self._fit_pca(X)

	def _fit_pca(self,X):
		pca = PCA(n_components=self.n_eofs, **self.method_opts)
		self.pca = pca
		# get projection coefficients (n_samples x n_eofs)
		self.U_ = pca.fit_transform(X)
		self.projections_ = self.U_	
		# get EOFs (each row is an EOF n_eofs x n_dim)
		self.eofs_ = pca.components_
		self.V_ = pca.components_
		# get singular values (square root of variances)
		self.S_ = pca.singular_values_
		# explained variance ratio for convergence plotting
		self.explained_variance_ = pca.explained_variance_
		self.explained_variance_ratio_ = pca.explained_variance_ratio_
		self.cumulative_explained_variance_ratio_ = np.cumsum(pca.explained_variance_ratio_)
		# compute total variance
		self.total_variance_ = (pca.singular_values_[0]**2/self.n_samples)/pca.explained_variance_ratio_[0]

		# compute varimax rotation is True
		if self.varimax == True:
			self.eofs_varimax_ = self._ortho_rotation(componentsT=self.eofs_.T)
			# also compute the explained variance
			X0 = X - np.mean(X,axis=0)
			U_varimax_ = np.dot(X0,self.eofs_varimax_.T)
			self.explained_variance_varimax_ = np.var(U_varimax_,axis=0)
			self.explained_variance_ratio_varimax_ = self.explained_variance_varimax_/self.total_variance_
			if "whiten" in self.method_opts:
				if self.method_opts['whiten'] == True:
					sigma = np.sqrt(self.explained_variance_varimax_)
					# scale each column of the projection by the stdev if whiten == True (default)
					U_varimax_ = np.dot(U_varimax_,np.diag(1./sigma))
			else:
				self.U_varimax_ = U_varimax_ # save projections
			self.projections_varimax_ = U_varimax_

	def _ortho_rotation(self,componentsT, method="varimax", tol=1e-8, max_iter=1000):
		"""Return rotated components (transpose).
		Here, componentsT are the transpose of the PCA components
		"""
		assert componentsT.shape[1] == self.n_eofs, "Input shape must be the transpose of the pca components_ vector."
		nrow, ncol = componentsT.shape
		rotation_matrix = np.eye(ncol)
		var = 0

		for _ in range(max_iter):
			comp_rot = np.dot(componentsT, rotation_matrix)
			if method == "varimax":
				tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
			elif method == "quartimax":
				tmp = 0
			u, s, v = np.linalg.svd(np.dot(componentsT.T, comp_rot ** 3 - tmp))
			rotation_matrix = np.dot(u, v)
			var_new = np.sum(s)
			if var != 0 and var_new < var * (1 + tol):
				break
			var = var_new

		self.rotation_matrix = rotation_matrix

		return np.dot(componentsT, rotation_matrix).T		
	