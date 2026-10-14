import json
import os
from collections import defaultdict

SYSTEM_PROMPT = "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn from streaming speech recognition, or 'no voice' when no speech is detected. Output exactly 'listen' when you should yield and keep listening; otherwise output a brief, natural speech fragment that can be spoken now. Do not repeat prior fragments. Continue an unfinished answer across turns when appropriate. Yield immediately for a clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."

DATA = {}

DATA["asr_noisy_repair"] = [
    # 0: conv_00219
    [
        ("I need to book tickets", "listen"),
        ("for three people to Chicago", "listen"),
        ("leaving at nine tomorrow", "listen"),
        ("no voice", "I can check train departures."),
        ("sorry I meant two people", "listen"),
        ("and departing at ten", "listen"),
        ("no voice", "Got it: two tickets at ten."),
        ("no voice", "Would you like morning or night?"),
        ("no voice", "listen")
    ],
    # 1: conv_00030
    [
        ("I want to order flowers", "listen"),
        ("a dozen yellow roses please", "listen"),
        ("for delivery on Thursday", "listen"),
        ("no voice", "I can arrange that floral order."),
        ("wait sorry I meant red", "listen"),
        ("and delivery on Friday afternoon", "listen"),
        ("no voice", "Updated: one dozen red roses Friday."),
        ("no voice", "Should I add a gift note?"),
        ("no voice", "listen")
    ],
    # 2: conv_00240
    [
        ("Schedule a dental checkup please", "listen"),
        ("for next Monday morning", "listen"),
        ("around eight thirty if possible", "listen"),
        ("no voice", "Looking at Monday morning dental slots."),
        ("actually I said Tuesday morning", "listen"),
        ("nine thirty works better", "listen"),
        ("no voice", "Got it: Tuesday at nine thirty."),
        ("no voice", "I will send the calendar invite."),
        ("no voice", "listen")
    ],
    # 3: conv_00009
    [
        ("Reserve a hotel room please", "listen"),
        ("with a king bed suite", "listen"),
        ("checking in on the fourteenth", "listen"),
        ("no voice", "I can reserve that king suite."),
        ("oops speech recognition misheard that", "listen"),
        ("two queen beds on fifteenth", "listen"),
        ("no voice", "Noted: two queen beds fifteenth."),
        ("no voice", "Do you require parking included?"),
        ("no voice", "listen")
    ],
    # 4: conv_00156
    [
        ("Add some fruit to cart", "listen"),
        ("two pounds of green apples", "listen"),
        ("and a bunch of bananas", "listen"),
        ("no voice", "Added apples and bananas to cart."),
        ("wait my mic cut out", "listen"),
        ("I said honeycrisp red apples", "listen"),
        ("no voice", "Swapped to red honeycrisp apples."),
        ("no voice", "Anything else for your grocery order?"),
        ("no voice", "listen")
    ],
    # 5: conv_00261
    [
        ("I need an economy car", "listen"),
        ("pickup at Denver airport terminal", "listen"),
        ("for three days total", "listen"),
        ("no voice", "Searching Denver airport economy rentals."),
        ("correction the mic heard wrong", "listen"),
        ("compact SUV for four days", "listen"),
        ("no voice", "Updated to compact SUV four days."),
        ("no voice", "Do you prefer all-wheel drive?"),
        ("no voice", "listen")
    ],
    # 6: conv_00534
    [
        ("Order sandwiches for the team", "listen"),
        ("about fifteen turkey wraps total", "listen"),
        ("delivered by twelve noon", "listen"),
        ("no voice", "Setting up fifteen turkey wraps noon."),
        ("sorry bad microphone reception there", "listen"),
        ("twenty vegetarian wraps by one", "listen"),
        ("no voice", "Got it: twenty vegetarian wraps one."),
        ("no voice", "Do you need drinks included?"),
        ("no voice", "listen")
    ],
    # 7: conv_00303
    [
        ("Place a hold on books", "listen"),
        ("for the biography of Hamilton", "listen"),
        ("at the central library branch", "listen"),
        ("no voice", "Finding the Hamilton biography downtown."),
        ("no no voice text misheard", "listen"),
        ("biography of Benjamin Franklin downtown", "listen"),
        ("no voice", "Placed hold on Franklin biography."),
        ("no voice", "You will get an email ready."),
        ("no voice", "listen")
    ],
    # 8: conv_00135
    [
        ("Book conference room B today", "listen"),
        ("from two to three PM", "listen"),
        ("for our product review meeting", "listen"),
        ("no voice", "Booking room B at two PM."),
        ("hold on audio glitch occurred", "listen"),
        ("room C from three to four", "listen"),
        ("no voice", "Booked room C three to four."),
        ("no voice", "Need projector screen setup enabled?"),
        ("no voice", "listen")
    ],
    # 9: conv_00114
    [
        ("Call a cab for tomorrow", "listen"),
        ("pickup at five in morning", "listen"),
        ("going to terminal two departures", "listen"),
        ("no voice", "Scheduling taxi five AM terminal two."),
        ("sorry voice typing mixed up", "listen"),
        ("terminal three at six AM", "listen"),
        ("no voice", "Rescheduled taxi six AM terminal three."),
        ("no voice", "Driver details will arrive shortly."),
        ("no voice", "listen")
    ],
    # 10: conv_00324
    [
        ("I want two large pizzas", "listen"),
        ("one pepperoni and one cheese", "listen"),
        ("crust should be thin crust", "listen"),
        ("no voice", "Adding two large thin crust pizzas."),
        ("correction speech recognition error here", "listen"),
        ("deep dish and extra mushrooms", "listen"),
        ("no voice", "Changed order to deep dish mushrooms."),
        ("no voice", "Estimated delivery time forty minutes."),
        ("no voice", "listen")
    ]
]

DATA["assistant_interrupted"] = [
    # 0: conv_00004
    [
        ("How do I learn web development", "listen"),
        ("if I have zero background", "listen"),
        ("no voice", "Start with simple HTML pages"),
        ("no voice", "and basic CSS styles first"),
        ("wait I already know HTML", "listen"),
        ("focus strictly on frontend frameworks", "listen"),
        ("no voice", "Then dive directly into React."),
        ("no voice", "Build small reusable component projects"),
        ("no voice", "to practice modern state management."),
        ("no voice", "listen")
    ],
    # 1: conv_00130
    [
        ("Give me a running plan", "listen"),
        ("to finish my first five-k", "listen"),
        ("no voice", "Begin with gentle walk-run intervals"),
        ("no voice", "three times every week consistently"),
        ("actually I run three miles already", "listen"),
        ("I want to run faster", "listen"),
        ("no voice", "Then introduce weekly interval sprints."),
        ("no voice", "Include tempo runs once weekly"),
        ("no voice", "to build sustained cardiovascular pace."),
        ("no voice", "listen")
    ],
    # 2: conv_00172
    [
        ("Recommend a reliable lightweight laptop", "listen"),
        ("for general office productivity tasks", "listen"),
        ("no voice", "Look at mid-range lightweight ultrabooks"),
        ("no voice", "with eight gigabytes of memory"),
        ("hold on I do heavy video editing", "listen"),
        ("so I need dedicated graphics", "listen"),
        ("no voice", "Then choose a powerful workstation."),
        ("no voice", "Look for thirty-two gigabytes RAM"),
        ("no voice", "and a high-performance discrete GPU."),
        ("no voice", "listen")
    ],
    # 3: conv_00067
    [
        ("How do I make vegetable soup", "listen"),
        ("hearty for a chilly evening", "listen"),
        ("no voice", "Start by sautéing diced onions"),
        ("no voice", "carrots celery and minced garlic"),
        ("wait I am using a pressure cooker", "listen"),
        ("so adjust cooking times please", "listen"),
        ("no voice", "Cook under pressure twelve minutes."),
        ("no voice", "Release pressure naturally before opening"),
        ("no voice", "for deeply rich tender vegetables."),
        ("no voice", "listen")
    ],
    # 4: conv_00382
    [
        ("What is the fastest way", "listen"),
        ("to start speaking basic Spanish", "listen"),
        ("no voice", "Master basic vocabulary flashcards first"),
        ("no voice", "and conjugate present tense verbs"),
        ("stop I already speak fluent Portuguese", "listen"),
        ("so I need false cognates", "listen"),
        ("no voice", "Focus on contrasting tricky cognates."),
        ("no voice", "Listen to native conversational podcasts"),
        ("no voice", "to tune ears to differences."),
        ("no voice", "listen")
    ],
    # 5: conv_00214
    [
        ("What houseplants survive low light", "listen"),
        ("in a basement bedroom environment", "listen"),
        ("no voice", "Snake plants and cast iron plants"),
        ("no voice", "tolerate deep shade quite well"),
        ("wait I actually have large grow lights", "listen"),
        ("so light level is high", "listen"),
        ("no voice", "Then consider fiddle leaf figs."),
        ("no voice", "Bright light supports tropical blooms"),
        ("no voice", "and faster vibrant lush growth."),
        ("no voice", "listen")
    ],
    # 6: conv_00319
    [
        ("How do I connect external speakers", "listen"),
        ("to a modern smart television", "listen"),
        ("no voice", "Use a standard optical audio cable"),
        ("no voice", "or three point five millimeter"),
        ("hold on my soundbar has HDMI eARC", "listen"),
        ("tell me about eARC connection", "listen"),
        ("no voice", "Connect directly via HDMI eARC port."),
        ("no voice", "Enable eARC pass-through in TV settings"),
        ("no voice", "for uncompressed surround sound quality."),
        ("no voice", "listen")
    ],
    # 7: conv_00256
    [
        ("How should I organize old photos", "listen"),
        ("scattered across multiple external drives", "listen"),
        ("no voice", "Consolidate every photo into folders"),
        ("no voice", "sorted by year and month"),
        ("wait I want automated face tagging", "listen"),
        ("tell me software that does that", "listen"),
        ("no voice", "Try Immich or DigiKam software."),
        ("no voice", "Both offer local facial recognition"),
        ("no voice", "without uploading to cloud providers."),
        ("no voice", "listen")
    ]
]

DATA["backchannel_story"] = [
    # 0: conv_00591
    [
        ("So I went hiking yesterday", "listen"),
        ("up near the mountain ridge", "listen"),
        ("and clouds rolled in suddenly", "listen"),
        ("no voice", "Oh wow, that sounds dramatic."),
        ("then we found a stone shelter", "listen"),
        ("and waited out the hailstorm", "listen"),
        ("no voice", "Glad you found safe cover."),
        ("we saw a rainbow right after", "listen"),
        ("no voice", "listen")
    ],
    # 1: conv_00297
    [
        ("I tried baking sourdough bread", "listen"),
        ("from a wild yeast starter", "listen"),
        ("and fermented it for thirty hours", "listen"),
        ("no voice", "That takes real patient dedication."),
        ("the crust turned golden brown", "listen"),
        ("with an amazingly airy crumb", "listen"),
        ("no voice", "Nothing beats homemade warm sourdough."),
        ("my roommates ate the whole loaf", "listen"),
        ("no voice", "listen")
    ],
    # 2: conv_00276
    [
        ("I visited the flea market", "listen"),
        ("early on Sunday morning downtown", "listen"),
        ("browsing through vintage vinyl records", "listen"),
        ("no voice", "Flea markets have great treasures."),
        ("and stumbled across an original pressing", "listen"),
        ("of Miles Davis Kind of Blue", "listen"),
        ("no voice", "What an incredible lucky find."),
        ("it plays with zero scratches", "listen"),
        ("no voice", "listen")
    ],
    # 3: conv_00486
    [
        ("My road trip hit a snag", "listen"),
        ("our front tire went flat", "listen"),
        ("in the middle of nowhere", "listen"),
        ("no voice", "Oh no, that sounds stressful."),
        ("but a friendly local mechanic stopped", "listen"),
        ("and helped us swap spares", "listen"),
        ("no voice", "Kind strangers make all difference."),
        ("we made it just in time", "listen"),
        ("no voice", "listen")
    ]
]

DATA["clarification_needed"] = [
    # 0: conv_00520
    [
        ("Can you help improve it", "listen"),
        ("no voice", "What would you like improved?"),
        ("my personal portfolio website", "listen"),
        ("no voice", "Do you want better visuals,"),
        ("no voice", "faster performance, or improved copy?"),
        ("the mobile navigation is clunky", "listen"),
        ("no voice", "Simplify the responsive hamburger menu"),
        ("no voice", "and enlarge tap target buttons."),
        ("no voice", "listen")
    ],
    # 1: conv_00268
    [
        ("Can you optimize this code", "listen"),
        ("no voice", "Which aspect should we optimize?"),
        ("the database query section", "listen"),
        ("no voice", "Is it running slowly,"),
        ("no voice", "or consuming too much memory?"),
        ("it takes five seconds on search", "listen"),
        ("no voice", "Add indexed columns to filters"),
        ("no voice", "and paginate large result sets."),
        ("no voice", "listen")
    ],
    # 2: conv_00373
    [
        ("Make this shorter for me", "listen"),
        ("no voice", "What document are we shortening?"),
        ("my cover letter draft", "listen"),
        ("no voice", "Do you need single paragraph,"),
        ("no voice", "bullet points, or half page?"),
        ("limit it to three concise paragraphs", "listen"),
        ("no voice", "Cut introductory filler phrases"),
        ("no voice", "and highlight two key achievements."),
        ("no voice", "listen")
    ],
    # 3: conv_00037
    [
        ("How do I fix this", "listen"),
        ("no voice", "What error are you seeing?"),
        ("my network connection dropped", "listen"),
        ("no voice", "Is it Wi-Fi disconnecting,"),
        ("no voice", "or an Ethernet cable issue?"),
        ("the Wi-Fi keeps dropping every hour", "listen"),
        ("no voice", "Restart your gateway router modem"),
        ("no voice", "and update wireless network drivers."),
        ("no voice", "listen")
    ],
    # 4: conv_00331
    [
        ("Can you polish this text", "listen"),
        ("no voice", "What kind of text is it?"),
        ("an email to my manager", "listen"),
        ("no voice", "Do you want professional tone,"),
        ("no voice", "casual brevity, or assertive persuasion?"),
        ("keep it polite and brief", "listen"),
        ("no voice", "State the core request upfront"),
        ("no voice", "and include specific deadline dates."),
        ("no voice", "listen")
    ],
    # 5: conv_00247
    [
        ("I need a gift recommendation", "listen"),
        ("no voice", "Who is the gift for?"),
        ("my college roommate", "listen"),
        ("no voice", "What are their favorite hobbies,"),
        ("no voice", "and what budget range preferred?"),
        ("loves coffee under fifty dollars", "listen"),
        ("no voice", "A ceramic pour-over dripper set"),
        ("no voice", "with fresh specialty roasted beans."),
        ("no voice", "listen")
    ],
    # 6: conv_00079
    [
        ("How should I set up", "listen"),
        ("no voice", "What setup are you planning?"),
        ("my home recording studio", "listen"),
        ("no voice", "Is it for spoken podcasting,"),
        ("no voice", "or multi-track music production?"),
        ("voiceover work in a small bedroom", "listen"),
        ("no voice", "Use dynamic cardioid microphone isolation"),
        ("no voice", "and hang acoustic damping blankets."),
        ("no voice", "listen")
    ],
    # 7: conv_00457
    [
        ("Can we clean this up", "listen"),
        ("no voice", "What needs cleaning up today?"),
        ("this shared spreadsheet file", "listen"),
        ("no voice", "Are there duplicate rows found,"),
        ("no voice", "broken formulas, or messy headers?"),
        ("there are lots of duplicate entries", "listen"),
        ("no voice", "Use conditional formatting highlight duplicates"),
        ("no voice", "then remove redundant rows cleanly."),
        ("no voice", "listen")
    ]
]

DATA["completion_then_yield"] = [
    # 0: conv_00057
    [
        ("What are three habits for better sleep", "listen"),
        ("when waking up tired constantly", "listen"),
        ("no voice", "Maintain a consistent sleep schedule,"),
        ("no voice", "dim bright screens before bed,"),
        ("no voice", "and keep the bedroom cool."),
        ("no voice", "listen"),
        ("thanks that is very practical", "listen"),
        ("no voice", "You are very welcome."),
        ("no voice", "listen")
    ],
    # 1: conv_00246
    [
        ("Give me three tips for public speaking", "listen"),
        ("to calm down stage fright", "listen"),
        ("no voice", "Practice deep diaphragmatic belly breathing,"),
        ("no voice", "know your opening sentence cold,"),
        ("no voice", "and make steady eye contact."),
        ("no voice", "listen"),
        ("appreciate the tips so much", "listen"),
        ("no voice", "Glad I could help out."),
        ("no voice", "listen")
    ],
    # 2: conv_00183
    [
        ("How can I organize my daily desk", "listen"),
        ("to keep clutter under control", "listen"),
        ("no voice", "Clear away unused coffee mugs,"),
        ("no voice", "bundle loose charging cables neatly,"),
        ("no voice", "and store paper in trays."),
        ("no voice", "listen"),
        ("that helps me start fresh", "listen"),
        ("no voice", "Enjoy your clean organized desk."),
        ("no voice", "listen")
    ],
    # 3: conv_00078
    [
        ("What are good snacks for road trips", "listen"),
        ("that do not require refrigeration", "listen"),
        ("no voice", "Pack roasted unsalted mixed nuts,"),
        ("no voice", "whole dried fruits or bananas,"),
        ("no voice", "and whole-grain crunchy pita chips."),
        ("no voice", "listen"),
        ("perfect list for our trip", "listen"),
        ("no voice", "Have a wonderful safe drive."),
        ("no voice", "listen")
    ],
    # 4: conv_00519
    [
        ("Name three quick stretches for back pain", "listen"),
        ("after sitting down all afternoon", "listen"),
        ("no voice", "Try gentle seated spinal twists,"),
        ("no voice", "cat-cow poses on hands knees,"),
        ("no voice", "and kneeling hip flexor stretches."),
        ("no voice", "listen"),
        ("thank you my back feels better", "listen"),
        ("no voice", "Always happy to help you."),
        ("no voice", "listen")
    ],
    # 5: conv_00288
    [
        ("How can I reduce food waste", "listen"),
        ("when grocery shopping every weekend", "listen"),
        ("no voice", "Audit your pantry before shopping,"),
        ("no voice", "plan meals around perishables first,"),
        ("no voice", "and freeze extra vegetable scraps."),
        ("no voice", "listen"),
        ("great ideas I will try them", "listen"),
        ("no voice", "Wonderful. Good luck meal prepping."),
        ("no voice", "listen")
    ]
]

DATA["disagreement_correction"] = [
    # 0: conv_00327
    [
        ("I think the conference starts Wednesday", "listen"),
        ("no voice", "I show it as Wednesday."),
        ("actually the badge says Thursday morning", "listen"),
        ("no voice", "Thanks for correcting that detail."),
        ("yes Thursday at eight AM", "listen"),
        ("no voice", "Thursday at eight AM recorded."),
        ("no voice", "listen")
    ],
    # 1: conv_00012
    [
        ("I thought Seattle was state capital", "listen"),
        ("no voice", "Many people assume that."),
        ("no Olympia is the capital city", "listen"),
        ("no voice", "You are right, it is Olympia."),
        ("Olympia has the state capitol", "listen"),
        ("no voice", "Correct, Olympia holds the legislature."),
        ("no voice", "listen")
    ],
    # 2: conv_00600
    [
        ("The concert venue holds ten thousand", "listen"),
        ("no voice", "That sounds like high capacity."),
        ("wait the ticket site says five thousand", "listen"),
        ("no voice", "Thanks for catching that mistake."),
        ("five thousand maximum seating capacity", "listen"),
        ("no voice", "Got it: five thousand seats total."),
        ("no voice", "listen")
    ],
    # 3: conv_00264
    [
        ("Python was released in ninety-nine", "listen"),
        ("no voice", "That sounds plausible at first."),
        ("actually Guido released it in ninety-one", "listen"),
        ("no voice", "Good catch, February ninety-one indeed."),
        ("yes nineteen ninety-one officially", "listen"),
        ("no voice", "Nineteen ninety-one is correct date."),
        ("no voice", "listen")
    ],
    # 4: conv_00117
    [
        ("The flight duration is five hours", "listen"),
        ("no voice", "I have five hours logged."),
        ("wait the itinerary says three hours", "listen"),
        ("no voice", "Thanks for double checking that."),
        ("three hours nonstop flight time", "listen"),
        ("no voice", "Three hours nonstop flight noted."),
        ("no voice", "listen")
    ],
    # 5: conv_00558
    [
        ("The lease agreement is six months", "listen"),
        ("no voice", "I recorded a six month lease."),
        ("no looking closely it is twelve months", "listen"),
        ("no voice", "Appreciate you pointing that out."),
        ("twelve month standard lease term", "listen"),
        ("no voice", "Twelve month lease term confirmed."),
        ("no voice", "listen")
    ]
]

DATA["emotional_support"] = [
    # 0: conv_00101
    [
        ("I feel so overwhelmed today", "listen"),
        ("with finals and project deadlines", "listen"),
        ("no voice", "I hear you, that is heavy."),
        ("I feel like I cannot finish", "listen"),
        ("no voice", "Take a slow deep breath."),
        ("maybe just help me tackle one", "listen"),
        ("no voice", "Pick one small urgent task"),
        ("no voice", "and we will break it down."),
        ("no voice", "listen")
    ],
    # 1: conv_00269
    [
        ("I got rejected from that job", "listen"),
        ("after four rounds of interviews", "listen"),
        ("no voice", "I am truly sorry to hear."),
        ("it stings after so much effort", "listen"),
        ("no voice", "That kind of rejection really hurts."),
        ("I need to bounce back somehow", "listen"),
        ("no voice", "Give yourself time to feel it"),
        ("no voice", "before jumping into next applications."),
        ("no voice", "listen")
    ],
    # 2: conv_00248
    [
        ("Everything went wrong at work today", "listen"),
        ("my presentation slides failed completely", "listen"),
        ("no voice", "That sounds so deeply frustrating."),
        ("I was so embarrassed in front of team", "listen"),
        ("no voice", "Tech glitches happen to everyone."),
        ("help me put things in perspective", "listen"),
        ("no voice", "People remember your core message"),
        ("no voice", "far more than slide hiccups."),
        ("no voice", "listen")
    ],
    # 3: conv_00164
    [
        ("I have been feeling lonely lately", "listen"),
        ("moving to this new unfamiliar city", "listen"),
        ("no voice", "Relocating alone can feel isolating."),
        ("it takes so long to make friends", "listen"),
        ("no voice", "Building community takes patient time."),
        ("maybe suggest a low-pressure social activity", "listen"),
        ("no voice", "Try joining a local run club"),
        ("no voice", "or a weekly boardgame meetup."),
        ("no voice", "listen")
    ]
]

DATA["filler_and_mhm"] = [
    # 0: conv_00203
    [
        ("So um I wanted to ask", "listen"),
        ("about fixing a leaky faucet", "listen"),
        ("no voice", "Mm-hm, I am listening."),
        ("I have some basic hand tools", "listen"),
        ("but never did plumbing repair before", "listen"),
        ("no voice", "That is very common."),
        ("I worry about water pressure leaks", "listen"),
        ("no voice", "Turn off shutoff valve first:"),
        ("no voice", "replace worn rubber washers inside."),
        ("no voice", "listen")
    ],
    # 1: conv_00539
    [
        ("Uh well I was wondering", "listen"),
        ("how to clean cast iron pans", "listen"),
        ("no voice", "Mm-hm, go ahead please."),
        ("some say never use any soap", "listen"),
        ("while others say mild soap works", "listen"),
        ("no voice", "Mild soap is safe."),
        ("I just want to avoid rust", "listen"),
        ("no voice", "Dry thoroughly on stovetop heat:"),
        ("no voice", "wipe lightly with vegetable oil."),
        ("no voice", "listen")
    ],
    # 2: conv_00581
    [
        ("Like you know I was thinking", "listen"),
        ("about adopting an older rescue dog", "listen"),
        ("no voice", "Mm-hm, I hear you."),
        ("I work eight hours each day", "listen"),
        ("and live in an apartment building", "listen"),
        ("no voice", "Senior dogs match well."),
        ("they need less intensive exercise", "listen"),
        ("no voice", "Provide predictable walking routine daily:"),
        ("no voice", "older rescues settle nicely."),
        ("no voice", "listen")
    ],
    # 3: conv_00329
    [
        ("Er so basically I need advice", "listen"),
        ("on learning to play acoustic guitar", "listen"),
        ("no voice", "Mm-hm, tell me more."),
        ("my fingertips hurt after ten minutes", "listen"),
        ("and barre chords seem impossible now", "listen"),
        ("no voice", "Calluses develop over time."),
        ("should I practice every single day", "listen"),
        ("no voice", "Practice fifteen focused minutes daily:"),
        ("no voice", "consistency builds finger strength."),
        ("no voice", "listen")
    ],
    # 4: conv_00161
    [
        ("Um well you see I want", "listen"),
        ("to start an herb garden indoors", "listen"),
        ("no voice", "Mm-hm, I am with you."),
        ("I have a small kitchen windowsill", "listen"),
        ("that gets morning direct sunlight", "listen"),
        ("no voice", "Morning sun is great."),
        ("which herbs grow easiest there", "listen"),
        ("no voice", "Basil mint and rosemary thrive:"),
        ("no voice", "ensure drainage pots prevent rotting."),
        ("no voice", "listen")
    ],
    # 5: conv_00266
    [
        ("So uh I was thinking about", "listen"),
        ("switching to a standing desk", "listen"),
        ("no voice", "Mm-hm, keep going please."),
        ("my lower back gets sore sitting", "listen"),
        ("all day during remote coding shifts", "listen"),
        ("no voice", "Posture fatigue is real."),
        ("how often should I stand up", "listen"),
        ("no voice", "Alternate thirty minutes standing upright:"),
        ("no voice", "with thirty minutes sitting comfortably."),
        ("no voice", "listen")
    ],
    # 6: conv_00119
    [
        ("Like um I need some help", "listen"),
        ("choosing a good wireless router", "listen"),
        ("no voice", "Mm-hm, I follow you."),
        ("our apartment has thick concrete walls", "listen"),
        ("and our connection drops in bedroom", "listen"),
        ("no voice", "Dense walls block signal."),
        ("would a mesh system help us", "listen"),
        ("no voice", "A two-node mesh system works:"),
        ("no voice", "place satellite midway near bedroom."),
        ("no voice", "listen")
    ]
]

DATA["hesitation_wait"] = [
    # 0: conv_00127
    [
        ("Well I was thinking of", "listen"),
        ("maybe moving to another city", "listen"),
        ("but I am not sure", "listen"),
        ("no voice", "That makes complete sense."),
        ("I mostly worry about", "listen"),
        ("leaving my friends and family", "listen"),
        ("no voice", "We can weigh the pros"),
        ("no voice", "against career opportunities you have."),
        ("no voice", "listen")
    ],
    # 1: conv_00001
    [
        ("Um I have been debating", "listen"),
        ("whether to buy an electric vehicle", "listen"),
        ("or stick with hybrid cars", "listen"),
        ("no voice", "Both have great advantages."),
        ("my daily commute is short", "listen"),
        ("but I take long roadtrips", "listen"),
        ("no voice", "Consider charging station availability"),
        ("no voice", "along your frequent holiday routes."),
        ("no voice", "listen")
    ],
    # 2: conv_00064
    [
        ("I kind of want to", "listen"),
        ("sign up for a marathon", "listen"),
        ("though it sounds really daunting", "listen"),
        ("no voice", "That is an ambitious goal."),
        ("I have never run beyond", "listen"),
        ("maybe six or eight miles", "listen"),
        ("no voice", "A gradual sixteen-week training plan"),
        ("no voice", "builds endurance without injury."),
        ("no voice", "listen")
    ],
    # 3: conv_00358
    [
        ("I am sort of hesitating", "listen"),
        ("on whether to adopt a kitten", "listen"),
        ("because I travel for work", "listen"),
        ("no voice", "Travel makes pet ownership tricky."),
        ("I do have friendly neighbors", "listen"),
        ("who offered to cat-sit occasionally", "listen"),
        ("no voice", "A bonded pair of cats"),
        ("no voice", "can keep each other company."),
        ("no voice", "listen")
    ],
    # 4: conv_00337
    [
        ("Well I was contemplating", "listen"),
        ("asking for a promotion soon", "listen"),
        ("at our next quarterly review", "listen"),
        ("no voice", "That is a proactive step."),
        ("I feel a little nervous", "listen"),
        ("bringing up salary compensation numbers", "listen"),
        ("no voice", "Document your quantified team contributions"),
        ("no voice", "to justify market rate adjustments."),
        ("no voice", "listen")
    ],
    # 5: conv_00421
    [
        ("I have been mulling over", "listen"),
        ("taking night classes in design", "listen"),
        ("on top of full-time work", "listen"),
        ("no voice", "Balancing both requires clear boundaries."),
        ("I worry about burning out", "listen"),
        ("by the middle of semester", "listen"),
        ("no voice", "Start with a single course"),
        ("no voice", "to evaluate your weekly workload."),
        ("no voice", "listen")
    ]
]

DATA["long_explanation_user"] = [
    # 0: conv_00013
    [
        ("I am trying to learn cooking", "listen"),
        ("but my small apartment kitchen lacks counter space", "listen"),
        ("and I have only two burner stoves", "listen"),
        ("plus very limited cabinet storage for pans", "listen"),
        ("so elaborate recipes feel completely impossible", "listen"),
        ("no voice", "One-pot meals work brilliantly here."),
        ("maybe sheet pan dinners in the oven", "listen"),
        ("no voice", "Sheet pan recipes minimize dishes,"),
        ("no voice", "save cramped kitchen counter space,"),
        ("no voice", "and cook everything evenly together."),
        ("no voice", "That keeps weeknight cooking enjoyable."),
        ("no voice", "listen")
    ],
    # 1: conv_00076
    [
        ("I have been trying to read books", "listen"),
        ("every single evening before going to bed", "listen"),
        ("but as soon as I pick up book", "listen"),
        ("I get distracted by notification chimes on phone", "listen"),
        ("and end up doomscrolling until late midnight", "listen"),
        ("no voice", "Phone habits are easily triggered."),
        ("maybe leave my phone in another room", "listen"),
        ("no voice", "Charge your phone outside bedroom,"),
        ("no voice", "place your physical book bedside,"),
        ("no voice", "and read ten pages peacefully."),
        ("no voice", "That protects your evening focus."),
        ("no voice", "listen")
    ]
]

DATA["misheard_word"] = [
    # 0: conv_00397
    [
        ("I need a recipe for fresh pear", "listen"),
        ("no voice", "Do you mean fresh pear fruit"),
        ("no voice", "or a pair of dishes?"),
        ("pear fruit sorry speech to text error", "listen"),
        ("no voice", "Got it: fresh pear recipe."),
        ("no voice", "You could make poached pears"),
        ("no voice", "in spiced honey vanilla syrup."),
        ("no voice", "listen")
    ],
    # 1: conv_00187
    [
        ("Look up the price of flour", "listen"),
        ("no voice", "Do you mean baking flour"),
        ("no voice", "or fresh cut flower bouquets?"),
        ("baking flour the mic got confused", "listen"),
        ("no voice", "Understood: wheat baking flour prices."),
        ("no voice", "A five-pound bag averages three"),
        ("no voice", "to four dollars at supermarkets."),
        ("no voice", "listen")
    ],
    # 2: conv_00271
    [
        ("Can you buy some thyme", "listen"),
        ("no voice", "Do you mean herbal thyme"),
        ("no voice", "or purchasing clock time?"),
        ("the herb thyme voice typing typo", "listen"),
        ("no voice", "Got it: fresh thyme herb."),
        ("no voice", "Added fresh thyme sprigs list"),
        ("no voice", "for your upcoming grocery run."),
        ("no voice", "listen")
    ],
    # 3: conv_00166
    [
        ("Check the status of the plane", "listen"),
        ("no voice", "Do you mean airplane flight"),
        ("no voice", "or a carpenter wood plane?"),
        ("flight plane speech recognition slipped up", "listen"),
        ("no voice", "Tracking your incoming airline flight."),
        ("no voice", "It is on schedule arriving"),
        ("no voice", "at gate twelve on time."),
        ("no voice", "listen")
    ],
    # 4: conv_00019
    [
        ("I want to replace the brake", "listen"),
        ("no voice", "Do you mean bicycle brake"),
        ("no voice", "or taking a rest break?"),
        ("bicycle brake mic misheard the word", "listen"),
        ("no voice", "Got it: bike brake pads."),
        ("no voice", "Inspect the rubber pad wear"),
        ("no voice", "and tighten the steel cable."),
        ("no voice", "listen")
    ],
    # 5: conv_00418
    [
        ("Where can I find the sole", "listen"),
        ("no voice", "Do you mean shoe sole"),
        ("no voice", "or Dover sole fish fillet?"),
        ("shoe sole repair voice text slipped", "listen"),
        ("no voice", "Understood: shoe cobbler repair services."),
        ("no voice", "Local cobblers replace rubber soles"),
        ("no voice", "and extend boot lifespan years."),
        ("no voice", "listen")
    ]
]

DATA["multistep_task"] = [
    # 0: conv_00480
    [
        ("Help me build a personal budget", "listen"),
        ("starting this month from scratch", "listen"),
        ("no voice", "List your total monthly net income."),
        ("done next step after that", "listen"),
        ("no voice", "Subtract fixed costs like rent utilities."),
        ("okay got those subtracted now", "listen"),
        ("no voice", "Allocate remaining funds toward savings"),
        ("no voice", "and flexible weekly discretionary spending."),
        ("no voice", "listen")
    ],
    # 1: conv_00564
    [
        ("Guide me through setting up git", "listen"),
        ("on my brand new laptop", "listen"),
        ("no voice", "Install git and open your terminal."),
        ("installed what do I configure", "listen"),
        ("no voice", "Set user name and email config."),
        ("configured next step for github", "listen"),
        ("no voice", "Generate an SSH public key"),
        ("no voice", "and upload it to GitHub."),
        ("no voice", "listen")
    ],
    # 2: conv_00081
    [
        ("How do I repot this houseplant", "listen"),
        ("roots are coming out bottom", "listen"),
        ("no voice", "Choose a pot two inches wider."),
        ("got the pot what next", "listen"),
        ("no voice", "Add fresh potting soil at base."),
        ("plant is in the new soil", "listen"),
        ("no voice", "Fill soil around edges snugly"),
        ("no voice", "and water thoroughly until drained."),
        ("no voice", "listen")
    ],
    # 3: conv_00312
    [
        ("Help me prepare a presentation deck", "listen"),
        ("for our quarterly stakeholder meeting", "listen"),
        ("no voice", "Outline three main message points first."),
        ("points drafted what should follow", "listen"),
        ("no voice", "Add clear supporting charts and metrics."),
        ("charts added what is next", "listen"),
        ("no voice", "Write concise takeaway headline bullets"),
        ("no voice", "for every individual slide deck."),
        ("no voice", "listen")
    ],
    # 4: conv_00039
    [
        ("Walk me through making cold brew", "listen"),
        ("for hot summer morning drinks", "listen"),
        ("no voice", "Coarsely grind fresh coffee beans first."),
        ("ground what ratio of water", "listen"),
        ("no voice", "Combine four parts water to coffee."),
        ("mixed together what is next", "listen"),
        ("no voice", "Steep twelve hours in fridge"),
        ("no voice", "then filter through paper mesh."),
        ("no voice", "listen")
    ],
    # 5: conv_00270
    [
        ("Help me migrate my blog site", "listen"),
        ("to a fast static host", "listen"),
        ("no voice", "Export all markdown content files first."),
        ("content exported what comes next", "listen"),
        ("no voice", "Configure static site generator build settings."),
        ("build succeeds what is last", "listen"),
        ("no voice", "Point your DNS custom domain"),
        ("no voice", "to the static hosting provider."),
        ("no voice", "listen")
    ],
    # 6: conv_00396
    [
        ("How do I paint an interior room", "listen"),
        ("starting with clean bare walls", "listen"),
        ("no voice", "Wipe dust off and tape edges."),
        ("taped down what is next", "listen"),
        ("no voice", "Apply one coat of primer evenly."),
        ("primer dried what do I do", "listen"),
        ("no voice", "Roll two coats top paint"),
        ("no voice", "allowing proper dry time between."),
        ("no voice", "listen")
    ],
    # 7: conv_00123
    [
        ("Help me assemble this flatpack bookcase", "listen"),
        ("all parts laid on carpet", "listen"),
        ("no voice", "Separate screws and dowels into bowls."),
        ("hardware sorted what step one", "listen"),
        ("no voice", "Insert wooden dowels into shelf panels."),
        ("dowels inserted what is next", "listen"),
        ("no voice", "Attach side frame pieces securely"),
        ("no voice", "and nail back panel square."),
        ("no voice", "listen")
    ]
]

DATA["natural_conversation"] = [
    # 0: conv_00504
    [
        ("I tried making sourdough pancakes", "listen"),
        ("using leftover starter this morning", "listen"),
        ("no voice", "That sounds delicious."),
        ("they turned out really fluffy", "listen"),
        ("no voice", "The lactic acid adds fluffiness"),
        ("no voice", "and delicate tangy flavor."),
        ("we topped them with blueberries", "listen"),
        ("no voice", "Blueberries make the perfect pairing."),
        ("no voice", "listen")
    ],
    # 1: conv_00147
    [
        ("I visited a bonsai nursery today", "listen"),
        ("and saw trees fifty years old", "listen"),
        ("no voice", "Bonsai trees are living art."),
        ("the patience required is remarkable", "listen"),
        ("no voice", "Decades of careful wiring pruning"),
        ("no voice", "shape their miniature beauty."),
        ("I bought a small juniper tree", "listen"),
        ("no voice", "Keep it outdoors for health."),
        ("no voice", "listen")
    ],
    # 2: conv_00399
    [
        ("I watched meteor showers last night", "listen"),
        ("from our back porch hammock", "listen"),
        ("no voice", "A clear night for stargazing."),
        ("saw five bright shooting stars", "listen"),
        ("no voice", "Perseid meteors are always spectacular"),
        ("no voice", "against dark midnight skies."),
        ("it was so peaceful outside", "listen"),
        ("no voice", "Quiet stargazing nights refresh the spirit."),
        ("no voice", "listen")
    ],
    # 3: conv_00252
    [
        ("I finally finished a puzzle", "listen"),
        ("one thousand pieces of artwork", "listen"),
        ("no voice", "Congratulations on finishing it."),
        ("the last piece was under couch", "listen"),
        ("no voice", "Finding missing pieces is satisfying"),
        ("no voice", "after hours of searching."),
        ("now I might frame it", "listen"),
        ("no voice", "Framing preserves all your hard work."),
        ("no voice", "listen")
    ]
]

DATA["question_answer_chunks"] = [
    # 0: conv_00548
    [
        ("Can you explain", "listen"),
        ("how airplanes generate lift", "listen"),
        ("in aerodynamic terms", "listen"),
        ("no voice", "Air travels over curved wings"),
        ("no voice", "creating lower pressure above surface."),
        ("no voice", "Higher pressure underneath wings pushes"),
        ("no voice", "the aircraft up into air."),
        ("no voice", "listen")
    ],
    # 1: conv_00233
    [
        ("Can you explain", "listen"),
        ("why the ocean has tides", "listen"),
        ("in simple language", "listen"),
        ("no voice", "The moon exerts gravitational pull"),
        ("no voice", "on the Earth's ocean water."),
        ("no voice", "This causes ocean surface bulges"),
        ("no voice", "creating daily high low tides."),
        ("no voice", "listen")
    ],
    # 2: conv_00443
    [
        ("Tell me how", "listen"),
        ("noise canceling headphones work", "listen"),
        ("against loud engine hum", "listen"),
        ("no voice", "Microphones detect incoming external sounds"),
        ("no voice", "and generate inverted sound waves."),
        ("no voice", "These opposite waves cancel noise"),
        ("no voice", "before reaching your eardrums."),
        ("no voice", "listen")
    ],
    # 3: conv_00254
    [
        ("Can you describe", "listen"),
        ("how photosynthesis works in plants", "listen"),
        ("step by step", "listen"),
        ("no voice", "Chlorophyll absorbs red blue sunlight"),
        ("no voice", "converting water carbon dioxide into glucose."),
        ("no voice", "This process releases fresh oxygen"),
        ("no voice", "back into the surrounding atmosphere."),
        ("no voice", "listen")
    ],
    # 4: conv_00149
    [
        ("Explain to me", "listen"),
        ("why leaves change colors in autumn", "listen"),
        ("before falling off", "listen"),
        ("no voice", "Shorter daylight stops chlorophyll production"),
        ("no voice", "revealing carotenoids and anthocyanins."),
        ("no voice", "These pigments show vibrant yellows"),
        ("no voice", "oranges and deep autumn reds."),
        ("no voice", "listen")
    ],
    # 5: conv_00338
    [
        ("How does a", "listen"),
        ("thermos keep soup hot all day", "listen"),
        ("without any power", "listen"),
        ("no voice", "A double wall vacuum chamber"),
        ("no voice", "prevents heat transfer via conduction."),
        ("no voice", "Reflective silver lining bounces heat"),
        ("no voice", "radiation back inside the container."),
        ("no voice", "listen")
    ],
    # 6: conv_00086
    [
        ("Can you explain", "listen"),
        ("what causes a rainbow in sky", "listen"),
        ("after summer rain", "listen"),
        ("no voice", "Sunlight enters falling raindrops suspended"),
        ("no voice", "and refracts separating distinct colors."),
        ("no voice", "Total internal reflection redirects light"),
        ("no voice", "back toward the observer's eyes."),
        ("no voice", "listen")
    ],
    # 7: conv_00212
    [
        ("Tell me how", "listen"),
        ("microwave ovens heat cold food", "listen"),
        ("so quickly at home", "listen"),
        ("no voice", "A magnetron tube generates radio waves"),
        ("no voice", "that vibrate water molecules rapidly."),
        ("no voice", "This molecular friction produces heat"),
        ("no voice", "cooking food throughout evenly."),
        ("no voice", "listen")
    ]
]

DATA["short_acknowledgement"] = [
    # 0: conv_00448
    [
        ("I submitted my taxes online", "listen"),
        ("before the midnight deadline", "listen"),
        ("no voice", "Nicely done today."),
        ("it took all weekend though", "listen"),
        ("no voice", "Tax filings take serious effort."),
        ("glad it is done now", "listen"),
        ("no voice", "Agreed. Enjoy the relief."),
        ("no voice", "listen")
    ],
    # 1: conv_00574
    [
        ("I passed my driver road test", "listen"),
        ("on the very first attempt", "listen"),
        ("no voice", "Great job indeed."),
        ("I was so nervous parallel parking", "listen"),
        ("no voice", "Parallel parking causes real stress."),
        ("now I have my license", "listen"),
        ("no voice", "Drive safely out there."),
        ("no voice", "listen")
    ],
    # 2: conv_00427
    [
        ("We finished assembling the deck", "listen"),
        ("after three days of sawing", "listen"),
        ("no voice", "Impressive craftsmanship work."),
        ("every single muscle is sore", "listen"),
        ("no voice", "Carpentry is hard physical work."),
        ("time for a cold drink", "listen"),
        ("no voice", "Enjoy your well-earned rest."),
        ("no voice", "listen")
    ],
    # 3: conv_00490
    [
        ("I finally reached my savings goal", "listen"),
        ("for the emergency rainy fund", "listen"),
        ("no voice", "Huge milestone achieved."),
        ("it took eight months budgeting", "listen"),
        ("no voice", "Consistent financial discipline paid off."),
        ("peace of mind feels amazing", "listen"),
        ("no voice", "A great safety cushion."),
        ("no voice", "listen")
    ]
]

DATA["silence_then_resume"] = [
    # 0: conv_00367
    [
        ("I want to tell you about", "listen"),
        ("an unexpected job interview invitation", "listen"),
        ("no voice", "listen"),
        ("a recruiter reached out today", "listen"),
        ("for a senior engineering role", "listen"),
        ("no voice", "What did they tell you?"),
        ("they want an initial technical call", "listen"),
        ("no voice", "Review your recent architecture projects."),
        ("no voice", "listen")
    ],
    # 1: conv_00493
    [
        ("I was just thinking about", "listen"),
        ("my grandma's secret apple pie", "listen"),
        ("no voice", "listen"),
        ("I found her handwritten recipe card", "listen"),
        ("tucked inside an old cookbook", "listen"),
        ("no voice", "What ingredients did she use?"),
        ("she used granny smith apples nutmeg", "listen"),
        ("no voice", "Tart apples balance sweet spices."),
        ("no voice", "listen")
    ],
    # 2: conv_00598
    [
        ("I had an interesting idea", "listen"),
        ("for an open source project", "listen"),
        ("no voice", "listen"),
        ("a command line tool for logs", "listen"),
        ("that parses streaming json records", "listen"),
        ("no voice", "What features would it have?"),
        ("fast colorized filtering by key", "listen"),
        ("no voice", "Developers would love that utility."),
        ("no voice", "listen")
    ],
    # 3: conv_00451
    [
        ("Something funny happened earlier today", "listen"),
        ("down at the dog park", "listen"),
        ("no voice", "listen"),
        ("a golden retriever grabbed tennis balls", "listen"),
        ("three balls in mouth simultaneously", "listen"),
        ("no voice", "How did it carry all three?"),
        ("it refused to drop even one", "listen"),
        ("no voice", "Retrievers are truly dedicated creatures."),
        ("no voice", "listen")
    ]
]

DATA["urgent_chemical_safety"] = [
    # 0: conv_00131
    [
        ("I am cleaning the oven", "listen"),
        ("I have caustic lye cleaner", "listen"),
        ("and acidic vinegar mixed together", "Do not mix those."),
        ("no voice", "They cause violent exothermic bubbling."),
        ("I already splashed some on skin", "Flush with water now."),
        ("it is burning my arm", "Rinse under water immediately."),
        ("no voice", "Keep flushing for fifteen minutes."),
        ("no voice", "Seek urgent medical treatment promptly."),
        ("no voice", "listen")
    ],
    # 1: conv_00383
    [
        ("I am descaling the bathroom", "listen"),
        ("I have muriatic acid here", "listen"),
        ("and chlorine bleach bottle open", "Do not mix chemicals."),
        ("no voice", "Chlorine gas releases very rapidly."),
        ("I just poured both in drain", "Evacuate the bathroom immediately."),
        ("yellow cloud is forming up", "Get outside into air."),
        ("no voice", "Do not breathe the gas."),
        ("no voice", "Call emergency services or nine-one-one."),
        ("no voice", "listen")
    ],
    # 2: conv_00173
    [
        ("I am unclogging the sink", "listen"),
        ("poured sodium hydroxide pellets down", "listen"),
        ("and adding sulfuric drain opener", "Do not combine them."),
        ("no voice", "They generate extreme dangerous heat."),
        ("it is boiling and spitting hot", "Step back right now."),
        ("smoke is coming from drain", "Leave the kitchen area."),
        ("no voice", "Avoid toxic splashing fumes completely."),
        ("no voice", "Call poison control or emergency."),
        ("no voice", "listen")
    ],
    # 3: conv_00320
    [
        ("I am mixing pool chemicals", "listen"),
        ("I have granular pool chlorine", "listen"),
        ("and hydrochloric pool acid together", "Do not add acid."),
        ("no voice", "That combination produces lethal gas."),
        ("I already dumped them together", "Move upwind immediately."),
        ("it is sizzling green fumes", "Evacuate the pool area."),
        ("no voice", "Get to open fresh air."),
        ("no voice", "Alert emergency responders right away."),
        ("no voice", "listen")
    ],
    # 4: conv_00236
    [
        ("Cleaning mildew in my shower", "listen"),
        ("sprayed heavy concentrated bleach spray", "listen"),
        ("and rubbing alcohol bottle next", "Do not combine those."),
        ("no voice", "That reaction can create chloroform."),
        ("I already sprayed both around", "Leave the closed room."),
        ("feeling dizzy and lightheaded now", "Get into fresh air."),
        ("no voice", "Sit down in well-ventilated space."),
        ("no voice", "Call emergency medical services immediately."),
        ("no voice", "listen")
    ],
    # 5: conv_00404
    [
        ("I am polishing brass hardware", "listen"),
        ("I mixed hydrogen peroxide solution", "listen"),
        ("with strong concentrated vinegar liquid", "Do not mix those."),
        ("no voice", "That creates corrosive peracetic acid."),
        ("I spilled it on my hands", "Wash hands with soap."),
        ("skin is turning white stinging", "Flush under cold tap."),
        ("no voice", "Remove contaminated gloves or rings."),
        ("no voice", "Contact poison center for care."),
        ("no voice", "listen")
    ],
    # 6: conv_00047
    [
        ("Stripping old paint off furniture", "listen"),
        ("using methylene chloride paint stripper", "listen"),
        ("in an unventilated basement room", "Get out of basement."),
        ("no voice", "Vapors cause rapid carbon monoxide."),
        ("I feel very nauseous now", "Move outside right now."),
        ("stumbling on the stairs dizzy", "Breathe clean outdoor air."),
        ("no voice", "Lie down on your side."),
        ("no voice", "Call nine-one-one for emergency help."),
        ("no voice", "listen")
    ]
]

DATA["urgent_electrical_safety"] = [
    # 0: conv_00132
    [
        ("The light switch is smoking", "listen"),
        ("I might pull the plastic plate", "Do not touch it."),
        ("I can unscrew the wires", "Step away from switch."),
        ("no voice", "Move everyone away from wall."),
        ("no voice", "Shut off power at breaker."),
        ("no voice", "Call a licensed electrician promptly."),
        ("no voice", "listen")
    ],
    # 1: conv_00405
    [
        ("A power line fell down", "listen"),
        ("across my driveway near car", "Stay away from lines."),
        ("I could move it aside", "Do not approach wire."),
        ("no voice", "Downed lines remain energized dangerously."),
        ("no voice", "Keep thirty feet distance minimum."),
        ("no voice", "Call utility emergency dispatch immediately."),
        ("no voice", "listen")
    ],
    # 2: conv_00111
    [
        ("The extension cord is melting", "listen"),
        ("I might grab the plug", "Do not grab it."),
        ("I can yank the cord", "Do not touch cord."),
        ("no voice", "Stand clear of melted plastic."),
        ("no voice", "Cut breaker switch power safely."),
        ("no voice", "Keep fire extinguisher on standby."),
        ("no voice", "listen")
    ],
    # 3: conv_00447
    [
        ("Water flooded our basement floor", "listen"),
        ("near the submerged electrical panel", "Do not step inside."),
        ("I will wade to breaker", "Stay out of water."),
        ("no voice", "Standing water carries fatal voltage."),
        ("no voice", "Remain upstairs and stay safe."),
        ("no voice", "Call electric utility to disconnect."),
        ("no voice", "listen")
    ]
]

DATA["user_changes_topic"] = [
    # 0: conv_00029
    [
        ("Can you suggest dinner recipes", "listen"),
        ("something with salmon fillets", "listen"),
        ("no voice", "Pan sear salmon fillets"),
        ("no voice", "with lemon dill butter sauce."),
        ("actually can you check my flights", "listen"),
        ("flight four eighty-eight tomorrow", "listen"),
        ("no voice", "Checking flight four eighty-eight now."),
        ("is departure on schedule still", "listen"),
        ("no voice", "Departing terminal two on time,"),
        ("no voice", "gate forty-two at five PM."),
        ("no voice", "listen")
    ],
    # 1: conv_00512
    [
        ("How do I repot an orchid", "listen"),
        ("without damaging delicate roots", "listen"),
        ("no voice", "Use airy bark medium"),
        ("no voice", "and soak roots before potting."),
        ("wait forget that what weather", "listen"),
        ("in Boston this coming weekend", "listen"),
        ("no voice", "Looking at Boston weekend forecast."),
        ("will it rain on Saturday", "listen"),
        ("no voice", "Partly cloudy with mild breezes,"),
        ("no voice", "high around sixty-eight degrees Saturday."),
        ("no voice", "listen")
    ],
    # 2: conv_00407
    [
        ("Help me write an essay hook", "listen"),
        ("about renewable wind energy sources", "listen"),
        ("no voice", "Start with global wind turbine"),
        ("no voice", "growth over the last decade."),
        ("never mind help debug this python", "listen"),
        ("I have an index error", "listen"),
        ("no voice", "What line triggers index error?"),
        ("when accessing list at index zero", "listen"),
        ("no voice", "Check if list is empty"),
        ("no voice", "before indexing the first element."),
        ("no voice", "listen")
    ],
    # 3: conv_00281
    [
        ("Recommend a sci-fi book series", "listen"),
        ("with interstellar space exploration", "listen"),
        ("no voice", "Try The Expanse series"),
        ("no voice", "by James S A Corey."),
        ("actually tell me about espresso machines", "listen"),
        ("for beginners at home baristas", "listen"),
        ("no voice", "Begin with Breville Bambino machine."),
        ("does it steam milk well", "listen"),
        ("no voice", "It heats up in seconds"),
        ("no voice", "and steams silky microfoam milk."),
        ("no voice", "listen")
    ],
    # 4: conv_00554
    [
        ("What time does post office close", "listen"),
        ("the one on Main Street", "listen"),
        ("no voice", "Main Street branch closes"),
        ("no voice", "at five PM on weekdays."),
        ("wait tell me how make coldbrew", "listen"),
        ("with coarse ground dark roast", "listen"),
        ("no voice", "Steep coffee grounds in water"),
        ("for how many hours total", "listen"),
        ("no voice", "Twelve to sixteen hours refrigerated"),
        ("no voice", "then strain through paper filter."),
        ("no voice", "listen")
    ],
    # 5: conv_00428
    [
        ("How do I clean leather shoes", "listen"),
        ("scuffed up from wet snow", "listen"),
        ("no voice", "Wipe away salt stains gently"),
        ("no voice", "using diluted white vinegar solution."),
        ("forget that how make pie crust", "listen"),
        ("flaky with real chilled butter", "listen"),
        ("no voice", "Use ice-cold cubed butter."),
        ("should I use pastry cutter", "listen"),
        ("no voice", "Cut butter into pea-sized bits,"),
        ("no voice", "and do not overwork dough."),
        ("no voice", "listen")
    ],
    # 6: conv_00365
    [
        ("Suggest gifts for graphic designers", "listen"),
        ("under one hundred dollar budget", "listen"),
        ("no voice", "A Pantone color swatch book"),
        ("no voice", "or ergonomic vertical wireless mouse."),
        ("actually how fix squeaky door hinge", "listen"),
        ("in our upstairs master bedroom", "listen"),
        ("no voice", "Remove hinge pin slightly clean."),
        ("what lubricant works best there", "listen"),
        ("no voice", "Apply silicone spray or grease,"),
        ("no voice", "then tap pin back in."),
        ("no voice", "listen")
    ],
    # 7: conv_00239
    [
        ("What plants survive dry climate", "listen"),
        ("for a sunny outdoor patio", "listen"),
        ("no voice", "Agave succulents and lavender thrive"),
        ("no voice", "in full sun with drought."),
        ("wait tell me best workout routine", "listen"),
        ("for thirty minutes lunch break", "listen"),
        ("no voice", "Do bodyweight interval circuits."),
        ("pushups squats and lunges combined", "listen"),
        ("no voice", "Perform three sets each exercise"),
        ("no voice", "with minimal rest between sets."),
        ("no voice", "listen")
    ],
    # 8: conv_00533
    [
        ("How do I calibrate my monitor", "listen"),
        ("for accurate digital photo editing", "listen"),
        ("no voice", "Use a hardware colorimeter tool"),
        ("no voice", "and set six-five-hundred Kelvin."),
        ("actually can you summarize this email", "listen"),
        ("from our project lead today", "listen"),
        ("no voice", "Paste or read the email."),
        ("it says deadline moved Friday", "listen"),
        ("no voice", "The project deadline moved earlier:"),
        ("no voice", "adjust your deliverables schedule accordingly."),
        ("no voice", "listen")
    ]
]

DATA["user_interrupts_answer"] = [
    # 0: conv_00263
    [
        ("What is Docker containerization", "listen"),
        ("no voice", "Docker packages application code together"),
        ("no voice", "with all its system dependencies"),
        ("hold on what is an image", "listen"),
        ("no voice", "An image is a read-only template"),
        ("no voice", "used to instantiate runnable containers."),
        ("okay keep going please", "listen"),
        ("no voice", "Containers share the host operating kernel"),
        ("no voice", "making them lightweight and portable."),
        ("no voice", "listen")
    ],
    # 1: conv_00242
    [
        ("Explain how DNS resolution works", "listen"),
        ("no voice", "DNS translates human-readable domain names"),
        ("no voice", "into numerical IP network addresses"),
        ("wait what does IP mean", "listen"),
        ("no voice", "Internet Protocol: unique device identifiers"),
        ("no voice", "for routing network data packets."),
        ("got it please continue", "listen"),
        ("no voice", "Resolvers query authoritative name servers"),
        ("no voice", "to return matching target addresses."),
        ("no voice", "listen")
    ],
    # 2: conv_00368
    [
        ("What is machine learning overfitting", "listen"),
        ("no voice", "Overfitting occurs when a statistical model"),
        ("no voice", "memorizes noise in training data"),
        ("pause what causes high noise", "listen"),
        ("no voice", "Measurement errors or small sample sizes"),
        ("no voice", "introduce misleading patterns into datasets."),
        ("okay proceed with explanation", "listen"),
        ("no voice", "Regularization and cross-validation techniques prevent"),
        ("no voice", "models from memorizing idiosyncratic noise."),
        ("no voice", "listen")
    ],
    # 3: conv_00095
    [
        ("How does a relational database work", "listen"),
        ("no voice", "Relational databases organize structured data"),
        ("no voice", "into tables containing rows columns"),
        ("hold on what is a key", "listen"),
        ("no voice", "A primary key uniquely identifies rows,"),
        ("no voice", "while foreign keys link tables."),
        ("understood carry on now", "listen"),
        ("no voice", "SQL queries join related tables together"),
        ("no voice", "to retrieve interconnected relational records."),
        ("no voice", "listen")
    ],
    # 4: conv_00452
    [
        ("Explain how public key cryptography operates", "listen"),
        ("no voice", "Public key cryptography uses mathematically paired keys"),
        ("no voice", "to encrypt and decrypt information"),
        ("wait which key stays private", "listen"),
        ("no voice", "The private key stays strictly secret,"),
        ("no voice", "while the public key shares."),
        ("okay continue explaining encryption", "listen"),
        ("no voice", "Senders encrypt messages using public keys,"),
        ("no voice", "and only private keys decrypt."),
        ("no voice", "listen")
    ],
    # 5: conv_00284
    [
        ("What is continuous integration workflow", "listen"),
        ("no voice", "Continuous integration merges developer code changes"),
        ("no voice", "into a central shared repository"),
        ("pause what runs automated tests", "listen"),
        ("no voice", "CI build runners execute test suites"),
        ("no voice", "automatically on every committed push."),
        ("great continue the overview", "listen"),
        ("no voice", "Automated pipelines detect integration bugs early"),
        ("no voice", "before code reaches production deployments."),
        ("no voice", "listen")
    ],
    # 6: conv_00431
    [
        ("Explain how blockchain consensus works", "listen"),
        ("no voice", "Blockchain relies on distributed network nodes"),
        ("no voice", "agreeing on the transaction ledger"),
        ("wait what is proof of work", "listen"),
        ("no voice", "Proof of work requires computational puzzle-solving"),
        ("no voice", "to validate new transaction blocks."),
        ("got that please continue", "listen"),
        ("no voice", "Consensus mechanisms maintain decentralized ledger integrity"),
        ("no voice", "without trusting any central authority."),
        ("no voice", "listen")
    ]
]

DATA["user_requests_pause"] = [
    # 0: conv_00167
    [
        ("Explain how compounding interest works", "listen"),
        ("no voice", "Compound interest calculates earned interest"),
        ("no voice", "on both principal and accumulated"),
        ("wait slow down a second", "listen"),
        ("no voice", "Of course, taking time."),
        ("no voice", "You earn interest on interest,"),
        ("no voice", "so balances grow faster over time."),
        ("okay I follow you", "listen"),
        ("no voice", "listen")
    ],
    # 1: conv_00293
    [
        ("Tell me how neural networks train", "listen"),
        ("no voice", "Neural networks compute loss gradients backwards"),
        ("no voice", "using backpropagation algorithms through layers"),
        ("hold on slow down please", "listen"),
        ("no voice", "Sure thing, slowing down."),
        ("no voice", "The model adjusts its weights"),
        ("no voice", "to reduce prediction errors gradually."),
        ("okay that makes sense", "listen"),
        ("no voice", "listen")
    ],
    # 2: conv_00524
    [
        ("How do solar panels work", "listen"),
        ("no voice", "Photovoltaic cells absorb incoming photons"),
        ("no voice", "which knock electrons free generating"),
        ("wait pause a second please", "listen"),
        ("no voice", "No problem, standing by."),
        ("no voice", "Sunlight knocks electrons loose from silicon,"),
        ("no voice", "creating flowing direct electric current."),
        ("got it thank you", "listen"),
        ("no voice", "listen")
    ],
    # 3: conv_00335
    [
        ("Explain how fiber optic cables work", "listen"),
        ("no voice", "Fiber optic cables transmit data pulses"),
        ("no voice", "using rapid light wave reflections"),
        ("hold up pause a bit", "listen"),
        ("no voice", "Certainly, pausing for you."),
        ("no voice", "Light bounces through glass cores"),
        ("no voice", "carrying high-speed data across continents."),
        ("okay I am ready", "listen"),
        ("no voice", "listen")
    ],
    # 4: conv_00356
    [
        ("How does quantum computing work", "listen"),
        ("no voice", "Quantum computers use quantum bits called qubits"),
        ("no voice", "which exist in superposed states"),
        ("wait give me a second", "listen"),
        ("no voice", "Take all the time."),
        ("no voice", "Unlike classical bits of zero one,"),
        ("no voice", "qubits calculate multiple possibilities simultaneously."),
        ("alright that is clear", "listen"),
        ("no voice", "listen")
    ],
    # 5: conv_00251
    [
        ("Explain how GPS satellite tracking works", "listen"),
        ("no voice", "GPS receivers calculate precise distance vectors"),
        ("no voice", "from atomic clock satellite signals"),
        ("slow down just a minute", "listen"),
        ("no voice", "Happy to slow down."),
        ("no voice", "Four satellites pinpoint your exact location"),
        ("no voice", "through precise geometric trilateration calculations."),
        ("okay got the concept", "listen"),
        ("no voice", "listen")
    ],
    # 6: conv_00104
    [
        ("How does a jet turbine engine operate", "listen"),
        ("no voice", "Jet engines compress incoming intake air"),
        ("no voice", "mix fuel ignite expanding gases"),
        ("pause let me take notes", "listen"),
        ("no voice", "Take your time writing."),
        ("no voice", "Hot expanding gas rushes out back,"),
        ("no voice", "pushing the jet forward powerfully."),
        ("ready for next part", "listen"),
        ("no voice", "listen")
    ]
]


def generate():
    train_file = "/home/aditya/Desktop/STUDY/TalkAI/data/duplexcascade_synthetic/micro_turns_train.jsonl"
    out_file = "/home/aditya/Desktop/STUDY/TalkAI/data/duplexcascade_synthetic/train_part3.jsonl"

    with open(train_file, "r") as f:
        lines = [json.loads(line) for line in f]

    seen_content = set()
    duplicates = []
    for idx, item in enumerate(lines):
        turns_str = json.dumps(item.get("micro_turns"))
        if turns_str in seen_content:
            duplicates.append((idx, item["id"], item["scenario"]))
        else:
            seen_content.add(turns_str)

    part3_meta = duplicates[260:390]
    print(f"Loaded {len(part3_meta)} duplicate metadata records for Part 3.")

    scenario_ptrs = defaultdict(int)
    rewritten_items = []

    for idx, cid, scen in part3_meta:
        dialogue_idx = scenario_ptrs[scen]
        scenario_ptrs[scen] += 1
        
        dialogues_for_scen = DATA.get(scen, [])
        if dialogue_idx >= len(dialogues_for_scen):
            raise ValueError(f"Not enough dialogues for scenario {scen}: need index {dialogue_idx}")
            
        turns_pairs = dialogues_for_scen[dialogue_idx]
        micro_turns = [{"user": u, "assistant": a} for u, a in turns_pairs]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in turns_pairs:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        record = {
            "id": cid,
            "scenario": scen,
            "topic": f"{scen}_{cid}",
            "micro_turns": micro_turns,
            "messages": messages
        }
        rewritten_items.append(record)

    with open(out_file, "w") as f:
        for rec in rewritten_items:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully generated {len(rewritten_items)} items into {out_file}")


if __name__ == "__main__":
    generate()
