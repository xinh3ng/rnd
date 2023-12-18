
### Docker Build
```
version=  # package version
docker_image_tag=rnd:${version}
docker build -t ${docker_image_tag} .

# Pruning
docker system prune --all --force

# Running
docker run -p 6893:80 ${docker_image_tag}
(with env vars) $ docker run -e ENV_VARIABLE=<...> -p 6893:80 ${docker_image_tag}

# In the web browser
http://0.0.0.0:6893/docs

```

### Azure
```
docker tag ${docker_image_tag} xhpersonal.azurecr.io/$docker_image_tag

docker push xhpersonal.azurecr.io/$docker_image_tag

app_name=rnd
app_dns_name=rnd-dns

# Deploy
az container create --resource-group xh-personal --name $app_name --image xhpersonal.azurecr.io/$docker_image_tag --dns-name-label $app_dns_name --ports 80

# Get FQDN, e.g. rnd-dns.westus2.azurecontainer.io
az container show --resource-group xh-personal --name $app_name --query ipAddress.fqdn

az container show --resource-group xh-personal --name $app_name --query containers[0].instanceView.currentState.state

az container logs --resource-group xh-personal --name $app_name
```