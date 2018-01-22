.PHONY: clean, pull

clean: 
	rm -f *.pyc;
	cd stan; rm -f *.stan

pull:
	cd ~/anaconda2/lib/python2.7/site-packages/smc;
	git pull origin master;