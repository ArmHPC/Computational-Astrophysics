from astropy.io import fits
import matplotlib.pyplot as plt

fbs1447_fits = fits.open('data/fbs1447_cor.fits')

fbs1447_fits.info()

fbs1447 = fbs1447_fits[0].data

# fbs1447_fits['PRIMARY'].header

print(type(fbs1447))
print(fbs1447.shape)

plt.imshow(fbs1447, cmap='gray')
plt.colorbar()
plt.show()

