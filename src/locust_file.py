from locust import HttpUser, task, between, TaskSet
import random
import json

sentiment_sentences_positive = ["I love coding in python", "Today is very nice day", "This book is so good"]
sentiment_sentences_negative = ["James hates eating onions", "This painting is very ugly", "Food is very bad here"]
sentiment_sentences_neutral = ['This sentence is neutral', "The day after today is tomorow", "This is okay"]
summarization_texts = ["Pros . A LOT of space. This bag can hold a ton of stuff, and it has so many pockets!\
Many of the pockets have elastic around the top, to better hold things like bottles and sippy cups. \
The main compartment has pockets all the way around the inside, and the two small pockets on the outside on either end of the bag also have elastic that allows them to expand\
and hold the items in securely. . \
I easily fit everything that's in my current diaper bag, into this bag. Including: \
a full 100-count pack of wipes, a handful of disposable diapers, four cloth diapers, a travel-sized wet bag, a sippy cup, a couple of extra outfits, a few toys, \
the travel changing pad, an entire roll of flushable cloth diaper liners, two burp cloths, a 16-oz bottle of water and a handful of small toys. \
I put all of that in the main compartment, leaving the waterproof pocket (which I would probably use to put dirty cloth diapers in, while also using the travel wet bag), \
and the “mommy pocket” empty. I still had more than enough space to put things like my phone, wallet, etc. if I didn’t want to carry a purse and a diaper bag. . \
The main compartment zips on three sides, but it has pieces of fabric that stop it from flapping completely open. This is handy when rummaging around for something, \
but you don't want your stuff to slide out. And it has two zippers, so you don’t have to open it all the way if you don’t need to. . Convertible. \
You can carry this cross-body or over one shoulder with the long strap, as a backpack, or just by the top handle. The long shoulder strap is plenty long enough. \
And it has small straps that can be used to attach it to the back of a stroller. .", 
"The included travel changing pad is a good size, and larger than it seems in the photos. \
It's as wide as the bag is. . It's a good size - about the size of a backpack or small messenger bag. . \
The little feet on the bottom that keep the bag itself off of the ground are a nice feature, so you don't have to worry about setting it in anything questionable, \
or if you have to set the bag down someplace that's bound to be dirty (like a public restroom floor, an asphalt driveway, the sidewalk, etc). . \
It's a nice-looking bag. The coloring is nice, and I like the pattern. It's hard to find a patterned diaper bag that men wouldn't mind carrying around, too. \
My husband doesn't have a problem with it, and said he’d carry it. The gray fabric is nice, too. It has visual texture to it. It looks almost like gray denim. \
I like the fact that it isn't a solid color, which would show grime a lot more. Cons . The bag is heavy, even empty. That was my first thought when I took it out of the box. \
I thought maybe there was something left in it, but it was just the changing pad. I think it's all of the extra straps. \
You could take some of them off if you're not going to use them (all of them except for the handle on top are removable), \
but then what's the point of a convertible bag if you do that? . There's an imperfection in the pattern of the bag I received. \
It's not a big deal, but it bothers me now that I know it's there. . It's an odd shape for a backpack. \
It looks like a messenger bag, and I think it looks a little weird when worn as a backpack. But, again, not that big a deal. \
Wearing it as a backpack means it won’t slide forward and get in the way when you bend over, like crossbody bags do. . There are a LOT of straps. \
This is bound to happen with a convertible bag, though, so there really is no stopping it if you want the versatility of a bag that can be worn many ways. \
But I think the straps will get in the way if they’re all left on the bag all the time. ", 
"The cat is similar in anatomy to the other felid species: it has a strong flexible body, quick reflexes, sharp teeth, and retractable claws adapted to killing small prey.\
Its night vision and sense of smell are well developed. Cat communication includes vocalizations like meowing, purring, trilling, hissing, growling,\
and grunting as well as cat-specific body language. A predator that is most active at dawn and dusk (crepuscular), the cat is a solitary hunter but a social species.\
It can hear sounds too faint or too high in frequency for human ears, such as those made by mice and other small mammals. Cats also secrete and perceive pheromones."]

class SentimentTasks(TaskSet):
    @task
    def sentiment_positive(self):
        data = {
            "text" : random.choice(sentiment_sentences_positive)
        }
        self.client.post("/roberta_sentiment", data=json.dumps(data))
        print("pos")
        self.interrupt(reschedule=False)

    @task
    def sentiment_negative(self):
        data = {
            "text" : random.choice(sentiment_sentences_negative)
        }
        self.client.post("/roberta_sentiment", data=json.dumps(data))
        print("neg")
        self.interrupt(reschedule=False)

    @task
    def sentiment_neutral(self):
        data = {
            "text" : random.choice(sentiment_sentences_neutral)
        }
        self.client.post("/roberta_sentiment", data=json.dumps(data))
        print("neu")
        self.interrupt(reschedule=False)

class SummaryTasks(TaskSet):
    @task
    def summarization(self):
        data = {
            "text" : random.choice(summarization_texts)
        }
        self.client.post("/summarization", data=json.dumps(data))
        print("summarization")
        self.interrupt(reschedule=False)

class ClassificationTasks(TaskSet):
    @task
    def classification(self):
        self.client.post("/classification" + "?selection=MNB&num_of_rows=100&file_name=cleaned_data")
        print("classification")
        self.interrupt(reschedule=False)

class KMeansFromScratchTasks(TaskSet):
    @task
    def kmeansfs(self):
        self.client.post("/kmeans_from_scratch" + "?num_of_rows=100&num_of_max_clusters=5&file_name=cleaned_data")
        print("kmeans")
        self.interrupt(reschedule=False)

class MyUser(HttpUser):
    host = "http://127.0.0.1:8000"
    tasks = [SentimentTasks]
    wait_time = between(2, 5)
