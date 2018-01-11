vangogh:
	./main.py examples/content/mcmaster.jpg examples/style/vangogh.jpg -o vangogh_mcmaster.jpg \
	--beta 10000.0 --iterations 350

seurat:
		./main.py examples/content/mcmaster.jpg examples/style/seurat.jpg -o test_results.jpg \
		--beta 1000.0 --tv-weight 1e-6 --iterations 350

munch:
		./main.py examples/content/mcmaster.jpg examples/style/munch.jpg -o test_results.jpg \
		--beta 100000.0 --tv-weight 0 --alpha 1 --iterations 350 --avg-pooling --random-init

monet:
		./main.py examples/content/mcmaster.jpg examples/style/monet.jpg -o test_results.jpg \
		--style-weight 10000.0 --iterations 350 --avg-pooling

duchamp:
		./main.py examples/content/mcmaster.jpg examples/style/duchamp.jpg -o test_results.jpg \
		--style-weight 10000.0 --iterations 350 --avg-pooling

cezanne:
			./main.py examples/content/mcmaster.jpg examples/style/cezanne.jpg -o test_results.jpg \
			--style-weight 10000.0 --iterations 350 --avg-pooling
