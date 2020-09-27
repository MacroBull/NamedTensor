# NamedTensor
Tensor type (np.ndarray) with named dimensions and axes support

## Intro
How to represent a tensor with named dims and axes?

Suppose we have a 3x4 tensor, where:

* its dimension(ndim) is 2
* its shape is `(3, 4)`
* dim 0 has 3 axes
* dim 1 has 4 axes

If we assign names for dims and axes like:

* dim 0 for `Fruit`, with axes as `Apple, Banana, Coconut`
* dim 1 for `Size`, with axes as `Tiny, Small, Medium, Large`

A probability distribution tensor of a random chosen fruit can be written like:

							Size
			       	Tiny	Small	Medium	Large
			Apple 	0.000	0.006	0.196	0.222
	Fruit	Banana 	0.040	0.102	0.005	0.068
			Coconut 0.172	0.163	0.000	0.026

A NamedTensor could be accessed like:

	prob[Apple, Small] = 0.006
	prob[{'Size': Small, 'Fruit': Apple}] = 0.006
	prob[Coconut] =
				Size
		Tiny	Small	Medium	Large
		0.172	0.163	0.000	0.026

## NamedTensor
`FixedShapeNamedTensor` for NamedTensor where shape is immutable

creation:
```python
image = FixedShapeNamedTensor(
	dims=(
		('Batches', 4), 
		('Channels', 'RGB'),
		('V', range(-6, 6)),
		('U', range(-8, 8))
	))
# image = <FixedShapeNamedTensor shape=(4, 3, 12, 16), dims=('Batches', 'Channels', 'V', 'U')>
feature = FixedShapeNamedTensor(
	dims=(
		('Locations', tuple(zip(((3, 0), (-8, 2)), range(3)))),
		('Features', ('Objectness', 'dx', 'dy'))
	))
# feature = <FixedShapeNamedTensor shape=(2, 3), dims=('Locations', 'Features')>
```

access:
```python
image.dims['Channels'].B
# >>> 2
image[:, 'G', :0, :0]
# >>> <a (4, 6, 8) sub tensor>
feature[{'Features': 'Objectness', 'Locations': [(-8, 2), (3, 0)]}]
# >>> <a (2, ) sub tensor>
```

iteration
```python
list(feature.dims.flat)
# >>> 
#	[((3, 0), 'Objectness'),
# 	 ((3, 0), 'dx'),
# 	 ((3, 0), 'dy'),
# 	 ((-8, 2), 'Objectness'),
# 	 ((-8, 2), 'dx'),
# 	 ((-8, 2), 'dy')]
```

serialization
```python
image.tofile('/tmp/image.bin')
# cat /tmp/image.bin.ini
# 	dims=Batches,Channels,V,U
# 	dim=4
# 	dim=Channels(R=0, G=1, B=2)
# 	axes=-6,-5,-4,-3,-2,-1,0,1,2,3,4,5
# 	axes=-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7
# 	dtype=float64
```
