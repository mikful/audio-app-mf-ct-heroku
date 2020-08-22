# audio-app-mf-ct-heroku
Heroku deployment of mf ct audio app

# Git deployment steps

1. Create your heroku app and follow the steps in /deploy/heroku-git i.e.:
 * `$ heroku login`
 * `$ cd my-project/`
 * `$ heroku git:remote -a your-heroku-app-name`
 
2. Now follow the details regarding [Building Docker Images with heroku.yml](https://devcenter.heroku.com/articles/build-docker-images-heroku-yml#getting-started). i.e. Set the stack of your app to container: `heroku stack:set container`

3. Push your app to Heroku: `git push heroku master`

4. Connect your github repo directly for push deployments

