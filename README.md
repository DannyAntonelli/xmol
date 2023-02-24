# XMol

## How to deploy

### Staging

Use the staging configurations to test the SSL certificates from Let's Encrypt work. The [rate limits](https://letsencrypt.org/docs/rate-limits/) with this configuration are much higher than the ones of the production environment. Be mindful that this kind of staging certificate is not trusted publicly.

Create a `.env.staging` file with:

```bash
cp .env.staging.template .env.staging
```

Complete the fields in the newly created file with your information.

Build the containers:

```bash
sudo docker compose -f docker-compose.staging.yml build
```

Run the containers:

```bash
sudo docker compose -f docker-compose.staging.yml up
```

### Production

Create a `.env.prod` file with:

```bash
cp .env.prod.template .env.prod
```

Complete the fields in the newly created file with your information.

Build the containers:

```sh
sudo docker compose -f docker-compose.prod.yml build
```

Run the containers:

```sh
sudo docker compose -f docker-compose.prod.yml up
```