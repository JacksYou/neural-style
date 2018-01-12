vangogh:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
	                 --style-images examples/style/vangogh.jpg \
									 --style-weights 1 \
									 -o vangogh_mcmaster.jpg \
	                 --beta 10000.0 \
									 --iterations 350 \
									 --backend cuda \
									 --init content

vangogh2:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
									--style-images examples/style/vangogh2.jpg \
									--style-weights 1 \
									-o vangogh2_mcmaster.jpg \
									--beta 25000.0 \
									--iterations 350 \
									--backend cuda \
									--init content
vangogh_mix:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
									 --style-images examples/style/vangogh.jpg examples/style/vangogh2.jpg \
									 --style-weights 0.5 0.5 \
									 -o vangogh2_mcmaster.jpg \
									 --beta 15000.0 \
									 --iterations 350 \
									 --backend cuda \
									 --init content

picasso:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
									 --style-images examples/style/picasso.jpg \
									 --style-weights 1 \
									 -o picasso_mcmaster.jpg \
									 --beta 10000.0 \
									 --iterations 350 \
									 --backend cuda \
									 --init content 
seurat:
	./neuralstyle.py examples/content/mcmaster.jpg examples/style/seurat.jpg -o test_results.jpg \
	--Beta 1000.0 --Lambda 1e-6 --iterations 350

munch:
	./neuralstyle.py examples/content/mcmaster.jpg examples/style/munch.jpg -o test_results.jpg \
	--Beta 100000.0 --Lambda 0 --Alpha 1 --iterations 350 --init random

monet:
	./neuralstyle.py examples/content/mcmaster.jpg examples/style/monet.jpg -o test_results.jpg \
	--Beta 10000.0 --iterations 350

duchamp:
	./neuralstyle.py examples/content/mcmaster.jpg examples/style/duchamp.jpg -o test_results.jpg \
	--Beta 10000.0 --iterations 350

cezanne:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
	                 --style-image examples/style/cezanne.jpg \
									 -o test_results.jpg \
	                 --Beta 10000.0 \
									 --iterations 350
