DOCKER_VERBOSE_ARGS :=
ifneq ($(VERBOSE),)
	DOCKER_VERBOSE_ARGS := --progress=plain
endif


build-sifs: torch.Dockerfile | sifs
	docker build $(DOCKER_VERBOSE_ARGS) -f $< -t torch:agentformer .
	sudo singularity build sifs/torch.sif docker-daemon://torch:agentformer

sifs:
	mkdir -p sifs
