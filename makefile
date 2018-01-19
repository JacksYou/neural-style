vangogh_mcmaster:
	./neuralstyle.py \
	  --content-image examples/content/mcmaster.jpg \
	  --style-images examples/style/vangogh/starrynight_over_rhone.jpg \
		--style-weights 1 \
		-o vangogh_mcmaster.jpg \
	  --beta 10000.0 \
		--iterations 350 \
		--backend cuda \
		--init content

vangogh_mix_mcmaster:
	./neuralstyle.py \
	  --content-image examples/content/mcmaster.jpg \
		--style-images examples/style/starrynight_over_rhone.jpg \
								   examples/style/starrynight.jpg \
		--style-weights 0.5 0.5 \
		-o vangogh_mix_mcmaster.jpg \
		--beta 15000.0 \
		--iterations 350 \
		--backend cuda \
		--init content

picasso_mcmaster:
	./neuralstyle.py \
	  --content-image examples/content/mcmaster.jpg \
	  --style-images examples/style/picasso/figure.jpg \
	  --style-weights 1 \
	  -o picasso_mcmaster.jpg \
	  --beta 10000.0 \
	  --iterations 300 \
	  --backend cuda \
	  --init content

cezanne_fuji:
		./neuralstyle.py \
			 --content-image examples/content/fuji_blossoms.jpg \
			 --style-images examples/style/cezanne/mont_victoire_pine.jpg \
			 --style-weights 1 \
			 -o cezanne_fuji.jpg \
			 --lambda 0 \
			 --beta 10000.0 \
			 --iterations 400 \
			 --backend cuda \
			 --init content

monet_fuji:
	./neuralstyle.py \
		--content-image examples/content/fuji_blossoms.jpg \
		--style-images examples/style/monet/antibes.jpg \
		--style-weights 1 \
		-o monet_fuji.jpg \
		--beta 10000.0 \
		--iterations 400 \
		--backend cuda \
		--init content

vangogh_hardy:
		./neuralstyle.py \
				--content-image examples/content/tom_hardy2.jpg \
				--style-images examples/style/van_gogh/selfportrait.jpg \
				--style-weights 1 \
				-o vangogh_hardy.jpg \
				--beta 10000.0 \
				--iterations 300 \
				--backend cuda \
				--init content

snow_dog:
	./neuralstyle.py --content-image examples/content/petra.jpg \
									 --style-images examples/style/monet/road_to_giverny.jpg \
									 --style-weights 1 \
									 -o snow_dog.jpg \
									 --beta 100000.0 \
									 --alpha 1.0 \
									 --lambda 0 \
									 --iterations 300 \
									 --backend cuda \
									 --init content

munch_mcmaster:
	./neuralstyle.py examples/content/mcmaster.jpg \
									 examples/style/munch/the_scream.jpg \
									 -o munch_mcmaster.jpg \
	                 --beta 100000.0 \
									 --lambda 0 \
									 --alpha 1 \
									 --iterations 350 \
									 --init random

waterlilies_mcmaster:
	./neuralstyle.py examples/content/mcmaster.jpg \
	                 examples/style/monet/waterlilies.jpg
									 -o waterlilies_mcmaster.jpg \
	                 --beta 10000.0 \
									 --iterations 350

giverny_mcmaster_gardens:
	./neuralstyle.py --content-image examples/content/mcmaster_gardens.jpg \
									 --style-image examples/style/monet/garten_in_giverny.jpg \
									 --style-weights 1 \
									 -o giverny_mcmaster.jpg \
									 --beta 10000.0 \
									 --iterations 350 \
									 --backend cuda \
									 --init content

duchamp_mcmaster:
	./neuralstyle.py examples/content/mcmaster.jpg \
	                 examples/style/duchamp/staircase.jpg \
									 -o duchamp_mcmaster.jpg \
	                 --beta 10000.0 \
									 --iterations 350

cezanne_mcmaster:
	./neuralstyle.py --content-image examples/content/mcmaster.jpg \
	                 --style-image examples/style/cezanne.jpg \
									 -o test_results.jpg \
	                 --beta 10000.0 \
									 --iterations 350
