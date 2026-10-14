import json

# We will define a structured dictionary of conversations for all 130 items in part 1.
# Each entry will specify the exact micro_turns list: [{"user": ..., "assistant": ...}, ...]

CONVERSATIONS = {}

# 1. asr_noisy_repair (3 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00051"] = [
    {"user": "I need to order flowers", "assistant": "listen"},
    {"user": "for my mom on Tuesday", "assistant": "listen"},
    {"user": "maybe a dozen lilies", "assistant": "listen"},
    {"user": "no voice", "assistant": "I can help with floral delivery."},
    {"user": "sorry, I meant yellow roses", "assistant": "listen"},
    {"user": "and delivered on Thursday", "assistant": "listen"},
    {"user": "no voice", "assistant": "Understood: yellow roses for Thursday."},
    {"user": "no voice", "assistant": "What recipient address should I use?"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00198"] = [
    {"user": "Book two train tickets", "assistant": "listen"},
    {"user": "heading to Boston tomorrow", "assistant": "listen"},
    {"user": "in the morning around nine", "assistant": "listen"},
    {"user": "no voice", "assistant": "Looking up morning departures now."},
    {"user": "actually make that three tickets", "assistant": "listen"},
    {"user": "leaving after eleven instead", "assistant": "listen"},
    {"user": "no voice", "assistant": "Got it: three tickets after eleven."},
    {"user": "no voice", "assistant": "Should I select economy seats?"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00513"] = [
    {"user": "Schedule a dental cleaning", "assistant": "listen"},
    {"user": "with doctor Chen next Monday", "assistant": "listen"},
    {"user": "at two in the afternoon", "assistant": "listen"},
    {"user": "no voice", "assistant": "Checking open slots with Doctor Chen."},
    {"user": "wait, doctor Patel on Wednesday", "assistant": "listen"},
    {"user": "at four thirty instead", "assistant": "listen"},
    {"user": "no voice", "assistant": "Updated: Doctor Patel Wednesday at four thirty."},
    {"user": "no voice", "assistant": "Shall I send the appointment confirmation?"},
    {"user": "no voice", "assistant": "listen"},
]

# 2. assistant_interrupted (3 items):
# Template length: 10 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:listen

CONVERSATIONS["conv_00466"] = [
    {"user": "How do I start baking sourdough", "assistant": "listen"},
    {"user": "if I have never baked bread", "assistant": "listen"},
    {"user": "no voice", "assistant": "First culture a flour and water starter"},
    {"user": "no voice", "assistant": "over seven to ten days"},
    {"user": "wait, a neighbor gave me active starter", "assistant": "listen"},
    {"user": "so can I bake immediately", "assistant": "listen"},
    {"user": "no voice", "assistant": "Yes, you can skip culturing."},
    {"user": "no voice", "assistant": "Feed it once to peak volume"},
    {"user": "no voice", "assistant": "then mix your main dough."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00151"] = [
    {"user": "How should I learn Spanish guitar", "assistant": "listen"},
    {"user": "as a complete acoustic beginner", "assistant": "listen"},
    {"user": "no voice", "assistant": "Begin by learning finger placement"},
    {"user": "no voice", "assistant": "and basic nylon string plucking"},
    {"user": "hold on, I play electric guitar already", "assistant": "listen"},
    {"user": "so what specifically changes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Then focus on classical right-hand posture"},
    {"user": "no voice", "assistant": "and rest-stroke finger independence drills"},
    {"user": "no voice", "assistant": "rather than pick strumming."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00424"] = [
    {"user": "How do I build an iOS app", "assistant": "listen"},
    {"user": "without prior mobile development experience", "assistant": "listen"},
    {"user": "no voice", "assistant": "Download Xcode on your Mac"},
    {"user": "no voice", "assistant": "and follow Swift programming basics"},
    {"user": "wait, I already write React Native", "assistant": "listen"},
    {"user": "so I know cross platform concepts", "assistant": "listen"},
    {"user": "no voice", "assistant": "Then dive straight into SwiftUI views"},
    {"user": "no voice", "assistant": "and examine Apple state management bindings"},
    {"user": "no voice", "assistant": "to contrast with hooks."},
    {"user": "no voice", "assistant": "listen"},
]

# 3. backchannel_story (8 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:SPEAK | A:listen
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00024"] = [
    {"user": "So I went hiking yesterday", "assistant": "listen"},
    {"user": "up on the ridge trail", "assistant": "listen"},
    {"user": "when dense fog rolled in", "assistant": "listen"},
    {"user": "no voice", "assistant": "That sounds quite disorienting."},
    {"user": "but I followed the stone cairns", "assistant": "listen"},
    {"user": "and reached the summit shelter safely", "assistant": "listen"},
    {"user": "no voice", "assistant": "Glad you found the shelter."},
    {"user": "the hot tea there was amazing", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00066"] = [
    {"user": "I tried to bake croissants", "assistant": "listen"},
    {"user": "from scratch on Saturday morning", "assistant": "listen"},
    {"user": "and the butter melted through", "assistant": "listen"},
    {"user": "no voice", "assistant": "Lamination trouble can be discouraging."},
    {"user": "so I chilled the dough again", "assistant": "listen"},
    {"user": "and they puffed up beautifully anyway", "assistant": "listen"},
    {"user": "no voice", "assistant": "A wonderful save indeed."},
    {"user": "everyone at brunch loved them", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00234"] = [
    {"user": "My flight was diverted last night", "assistant": "listen"},
    {"user": "due to severe thunderstorms in Chicago", "assistant": "listen"},
    {"user": "we sat on the tarmac forever", "assistant": "listen"},
    {"user": "no voice", "assistant": "Tarmac delays are exhausting."},
    {"user": "then the airline gave meal vouchers", "assistant": "listen"},
    {"user": "and rebooked me in first class", "assistant": "listen"},
    {"user": "no voice", "assistant": "That is a great compensation."},
    {"user": "I slept comfortably the whole flight", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00339"] = [
    {"user": "I entered my very first 5k", "assistant": "listen"},
    {"user": "downtown this morning with friends", "assistant": "listen"},
    {"user": "and my shoelace untied immediately", "assistant": "listen"},
    {"user": "no voice", "assistant": "Right off the starting line!"},
    {"user": "I tied it quick and caught up", "assistant": "listen"},
    {"user": "even beat my goal by two minutes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Impressive recovery and finish."},
    {"user": "we celebrated with giant pancakes afterward", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00255"] = [
    {"user": "I planted heirloom tomato seeds", "assistant": "listen"},
    {"user": "in small starter pots weeks ago", "assistant": "listen"},
    {"user": "and nothing sprouted for twenty days", "assistant": "listen"},
    {"user": "no voice", "assistant": "The suspense can be tough."},
    {"user": "then I added a seedling heating pad", "assistant": "listen"},
    {"user": "and tiny green sprouts appeared overnight", "assistant": "listen"},
    {"user": "no voice", "assistant": "Warm soil makes a huge difference."},
    {"user": "now all twelve cups have seedlings", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00171"] = [
    {"user": "I adopted a rescue kitten yesterday", "assistant": "listen"},
    {"user": "from the local animal shelter", "assistant": "listen"},
    {"user": "she hid behind the sofa for hours", "assistant": "listen"},
    {"user": "no voice", "assistant": "Moving into a new home is scary."},
    {"user": "I offered some warm wet food", "assistant": "listen"},
    {"user": "and she crawled out purring loudly", "assistant": "listen"},
    {"user": "no voice", "assistant": "That is so heartwarming."},
    {"user": "she fell asleep right on my lap", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00528"] = [
    {"user": "I went out stargazing last night", "assistant": "listen"},
    {"user": "with a cheap beginner telescope", "assistant": "listen"},
    {"user": "and could only see blurry white dots", "assistant": "listen"},
    {"user": "no voice", "assistant": "Focusing small optics takes patience."},
    {"user": "then I adjusted the fine focus knob", "assistant": "listen"},
    {"user": "and saw the rings of Saturn clearly", "assistant": "listen"},
    {"user": "no voice", "assistant": "Seeing planetary rings is magical."},
    {"user": "I stayed outside until two in the morning", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00129"] = [
    {"user": "I attended a pottery workshop today", "assistant": "listen"},
    {"user": "learning to center clay on the wheel", "assistant": "listen"},
    {"user": "my first three bowls collapsed completely", "assistant": "listen"},
    {"user": "no voice", "assistant": "Wheel throwing has a steep curve."},
    {"user": "the instructor adjusted my hand stance", "assistant": "listen"},
    {"user": "and I made a neat little vase", "assistant": "listen"},
    {"user": "no voice", "assistant": "Terrific progress in one session."},
    {"user": "it goes into the kiln next week", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 1 loaded: ASR, Assistant Interrupted, Backchannel Story")

# 4. clarification_needed (5 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:no voice | A:SPEAK
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:no voice | A:SPEAK
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00163"] = [
    {"user": "Help me pick an outfit", "assistant": "listen"},
    {"user": "no voice", "assistant": "What event are you dressing for?"},
    {"user": "an evening banquet dinner", "assistant": "listen"},
    {"user": "no voice", "assistant": "Is it formal black-tie,"},
    {"user": "no voice", "assistant": "or business cocktail attire?"},
    {"user": "business cocktail at an art museum", "assistant": "listen"},
    {"user": "no voice", "assistant": "A tailored dark blazer with trousers"},
    {"user": "no voice", "assistant": "or a sleek midi dress works."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00142"] = [
    {"user": "Can you recommend a laptop", "assistant": "listen"},
    {"user": "no voice", "assistant": "What primary tasks will you do?"},
    {"user": "video editing and rendering", "assistant": "listen"},
    {"user": "no voice", "assistant": "Do you prefer macOS"},
    {"user": "no voice", "assistant": "or a Windows workstation?"},
    {"user": "Windows with a dedicated GPU", "assistant": "listen"},
    {"user": "no voice", "assistant": "Look for at least thirty-two gigs"},
    {"user": "no voice", "assistant": "and a high-tier RTX graphics card."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00478"] = [
    {"user": "Suggest a birthday present", "assistant": "listen"},
    {"user": "no voice", "assistant": "Who is the special gift for?"},
    {"user": "my teenage nephew who reads", "assistant": "listen"},
    {"user": "no voice", "assistant": "Does he like fantasy adventures"},
    {"user": "no voice", "assistant": "or science fiction thrillers?"},
    {"user": "epic sci-fi worldbuilding novels", "assistant": "listen"},
    {"user": "no voice", "assistant": "Consider Frank Herbert's Dune boxed set"},
    {"user": "no voice", "assistant": "or Liu Cixin's Three-Body trilogy."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00121"] = [
    {"user": "Can you design a workout", "assistant": "listen"},
    {"user": "no voice", "assistant": "What is your main fitness objective?"},
    {"user": "building upper body strength", "assistant": "listen"},
    {"user": "no voice", "assistant": "Do you train at a gym"},
    {"user": "no voice", "assistant": "or use bodyweight at home?"},
    {"user": "home using pullup bar and dumbbells", "assistant": "listen"},
    {"user": "no voice", "assistant": "Pair pullups and dumbbell overhead presses"},
    {"user": "no voice", "assistant": "with pushups and bent-over rows."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00016"] = [
    {"user": "I want to cook fish", "assistant": "listen"},
    {"user": "no voice", "assistant": "What type of fish fillet?"},
    {"user": "fresh Atlantic salmon fillets", "assistant": "listen"},
    {"user": "no voice", "assistant": "Would you like pan searing"},
    {"user": "no voice", "assistant": "or oven roasting with herbs?"},
    {"user": "pan searing for crispy skin", "assistant": "listen"},
    {"user": "no voice", "assistant": "Pat skin completely dry first,"},
    {"user": "no voice", "assistant": "then sear skin-down on medium high."},
    {"user": "no voice", "assistant": "listen"},
]

# 5. completion_then_yield (6 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:no voice | A:SPEAK
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:listen
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00414"] = [
    {"user": "How can I improve sleep quality", "assistant": "listen"},
    {"user": "on stressful workdays", "assistant": "listen"},
    {"user": "no voice", "assistant": "Keep a consistent bed schedule,"},
    {"user": "no voice", "assistant": "dim screens thirty minutes before,"},
    {"user": "no voice", "assistant": "and keep your room cool."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "thanks, that makes sense", "assistant": "listen"},
    {"user": "no voice", "assistant": "Rest well tonight."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00435"] = [
    {"user": "What spices go well together", "assistant": "listen"},
    {"user": "in roasted butternut squash soup", "assistant": "listen"},
    {"user": "no voice", "assistant": "Combine freshly grated nutmeg,"},
    {"user": "no voice", "assistant": "a pinch of ground cinnamon,"},
    {"user": "no voice", "assistant": "and fresh garden thyme leaves."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "perfect, I will add those", "assistant": "listen"},
    {"user": "no voice", "assistant": "Enjoy the homemade soup."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00309"] = [
    {"user": "Give me three good habits", "assistant": "listen"},
    {"user": "to maintain dental health", "assistant": "listen"},
    {"user": "no voice", "assistant": "Brush gently for two minutes,"},
    {"user": "no voice", "assistant": "floss daily between tight teeth,"},
    {"user": "no voice", "assistant": "and rinse thoroughly after meals."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "appreciate the quick tips", "assistant": "listen"},
    {"user": "no voice", "assistant": "Glad to be helpful."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00393"] = [
    {"user": "How do I water indoor succulents", "assistant": "listen"},
    {"user": "during winter months", "assistant": "listen"},
    {"user": "no voice", "assistant": "Water only when soil dries,"},
    {"user": "no voice", "assistant": "soak until water drains out,"},
    {"user": "no voice", "assistant": "and avoid splashing fleshy leaves."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "thank you, very clear", "assistant": "listen"},
    {"user": "no voice", "assistant": "Happy indoor gardening."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00225"] = [
    {"user": "What helps prevent screen glare", "assistant": "listen"},
    {"user": "in a bright home office", "assistant": "listen"},
    {"user": "no voice", "assistant": "Reposition monitor away from windows,"},
    {"user": "no voice", "assistant": "lower ambient room lighting levels,"},
    {"user": "no voice", "assistant": "and attach an anti-glare hood."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "that will help my eyes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take care of your eyes."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00561"] = [
    {"user": "Suggest three simple cardio exercises", "assistant": "listen"},
    {"user": "that require zero equipment", "assistant": "listen"},
    {"user": "no voice", "assistant": "Do rhythmic jumping jacks,"},
    {"user": "no voice", "assistant": "brisk high-knees in place,"},
    {"user": "no voice", "assistant": "and mountain climbers on floor."},
    {"user": "no voice", "assistant": "listen"},
    {"user": "thanks, starting that now", "assistant": "listen"},
    {"user": "no voice", "assistant": "Have a great workout."},
    {"user": "no voice", "assistant": "listen"},
]

# 6. disagreement_correction (3 items):
# Template length: 7 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:no voice | A:SPEAK
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:no voice | A:SPEAK
# 6: U:no voice | A:listen

CONVERSATIONS["conv_00432"] = [
    {"user": "I thought the museum opened at nine", "assistant": "listen"},
    {"user": "no voice", "assistant": "The schedule states nine a.m."},
    {"user": "actually their website lists ten a.m.", "assistant": "listen"},
    {"user": "no voice", "assistant": "Thank you for the update."},
    {"user": "and closed on Mondays", "assistant": "listen"},
    {"user": "no voice", "assistant": "Ten a.m. and closed Mondays."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00243"] = [
    {"user": "The flight was departing from Gate B4", "assistant": "listen"},
    {"user": "no voice", "assistant": "Boarding was listed at B4."},
    {"user": "the board just switched to Gate C12", "assistant": "listen"},
    {"user": "no voice", "assistant": "Good catch on the gate change."},
    {"user": "boarding starts in ten minutes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Head over to C12 now."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00222"] = [
    {"user": "I believe the package was three pounds", "assistant": "listen"},
    {"user": "no voice", "assistant": "Recorded weight was three pounds."},
    {"user": "the postal scale reads five pounds", "assistant": "listen"},
    {"user": "no voice", "assistant": "Correcting weight to five pounds."},
    {"user": "so postage will be higher", "assistant": "listen"},
    {"user": "no voice", "assistant": "Adjusting the shipping rate now."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 2 loaded: clarification_needed, completion_then_yield, disagreement_correction")

# 7. emotional_support (7 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00437"] = [
    {"user": "I failed my certification exam today", "assistant": "listen"},
    {"user": "and I feel completely crushed", "assistant": "listen"},
    {"user": "no voice", "assistant": "That is really hard news."},
    {"user": "I worked so hard for months", "assistant": "listen"},
    {"user": "no voice", "assistant": "Your preparation still has value."},
    {"user": "can we review the weakest section", "assistant": "listen"},
    {"user": "no voice", "assistant": "Certainly. Which domain showed the"},
    {"user": "no voice", "assistant": "biggest gap in your scores?"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00038"] = [
    {"user": "My closest friend is moving away", "assistant": "listen"},
    {"user": "across the country next week", "assistant": "listen"},
    {"user": "no voice", "assistant": "Goodbyes to close friends hurt."},
    {"user": "the neighborhood will feel so empty", "assistant": "listen"},
    {"user": "no voice", "assistant": "Strong friendships survive long distances."},
    {"user": "help me think of a farewell gift", "assistant": "listen"},
    {"user": "no voice", "assistant": "What shared memory or inside"},
    {"user": "no voice", "assistant": "joke means the most?"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00311"] = [
    {"user": "I made a big mistake at work", "assistant": "listen"},
    {"user": "and deleted a production dataset", "assistant": "listen"},
    {"user": "no voice", "assistant": "Accidents happen in production systems."},
    {"user": "I am terrified to tell the manager", "assistant": "listen"},
    {"user": "no voice", "assistant": "Calm honesty builds engineering trust."},
    {"user": "help me draft a message to the team", "assistant": "listen"},
    {"user": "no voice", "assistant": "State the scope of impact"},
    {"user": "no voice", "assistant": "and your backup restoration steps."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00122"] = [
    {"user": "I feel burned out from work", "assistant": "listen"},
    {"user": "and cannot focus on anything", "assistant": "listen"},
    {"user": "no voice", "assistant": "Prolonged exhaustion drains cognitive energy."},
    {"user": "even minor emails feel impossible", "assistant": "listen"},
    {"user": "no voice", "assistant": "Give yourself permission to pause."},
    {"user": "how should I wind down tonight", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take a warm shower now"},
    {"user": "no voice", "assistant": "and step away from screens."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00416"] = [
    {"user": "My manuscript got rejected again", "assistant": "listen"},
    {"user": "by another literary journal today", "assistant": "listen"},
    {"user": "no voice", "assistant": "Creative rejections are deeply discouraging."},
    {"user": "maybe I should stop writing", "assistant": "listen"},
    {"user": "no voice", "assistant": "Many celebrated authors were rejected."},
    {"user": "help me read the reviewer notes objectively", "assistant": "listen"},
    {"user": "no voice", "assistant": "Share one specific note that"},
    {"user": "no voice", "assistant": "felt most constructive to you."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00479"] = [
    {"user": "I have to give a eulogy tomorrow", "assistant": "listen"},
    {"user": "and I keep breaking down crying", "assistant": "listen"},
    {"user": "no voice", "assistant": "Grief makes speaking publicly difficult."},
    {"user": "I want to honor their memory properly", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sincere emotion touches everyone listening."},
    {"user": "how do I pace myself up there", "assistant": "listen"},
    {"user": "no voice", "assistant": "Pause whenever tears well up"},
    {"user": "no voice", "assistant": "and breathe before continuing quietly."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00017"] = [
    {"user": "My beloved dog is very sick", "assistant": "listen"},
    {"user": "and the vet gives a poor prognosis", "assistant": "listen"},
    {"user": "no voice", "assistant": "Facing pet illness brings heartache."},
    {"user": "I am struggling with tough choices", "assistant": "listen"},
    {"user": "no voice", "assistant": "Focus on his everyday comfort."},
    {"user": "what questions should I ask the vet", "assistant": "listen"},
    {"user": "no voice", "assistant": "Ask about his pain levels"},
    {"user": "no voice", "assistant": "and palliative care quality markers."},
    {"user": "no voice", "assistant": "listen"},
]

# 8. filler_and_mhm (3 items):
# Template length: 10 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:SPEAK | A:listen
# 5: U:no voice | A:SPEAK
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:listen

CONVERSATIONS["conv_00224"] = [
    {"user": "So um I wanted to check", "assistant": "listen"},
    {"user": "about investing for retirement early", "assistant": "listen"},
    {"user": "no voice", "assistant": "I am listening."},
    {"user": "I opened a Roth IRA account", "assistant": "listen"},
    {"user": "but I do not know index funds", "assistant": "listen"},
    {"user": "no voice", "assistant": "That is normal initially."},
    {"user": "the ticker symbols confuse me", "assistant": "listen"},
    {"user": "no voice", "assistant": "Start with broad market funds"},
    {"user": "no voice", "assistant": "like low-cost total index portfolios."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00182"] = [
    {"user": "Uh so I was wondering", "assistant": "listen"},
    {"user": "about setting up home audio speakers", "assistant": "listen"},
    {"user": "no voice", "assistant": "Go right ahead."},
    {"user": "I have an old stereo receiver", "assistant": "listen"},
    {"user": "and bought two bookshelf speakers yesterday", "assistant": "listen"},
    {"user": "no voice", "assistant": "Good hardware foundation."},
    {"user": "I am confused by speaker impedance", "assistant": "listen"},
    {"user": "no voice", "assistant": "Match the speaker ohm rating"},
    {"user": "no voice", "assistant": "to your receiver output range."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00434"] = [
    {"user": "Well um I need advice", "assistant": "listen"},
    {"user": "on starting indoor vegetable gardening", "assistant": "listen"},
    {"user": "no voice", "assistant": "I hear you."},
    {"user": "my apartment has north-facing windows", "assistant": "listen"},
    {"user": "so natural sunlight is pretty scarce", "assistant": "listen"},
    {"user": "no voice", "assistant": "A very common challenge."},
    {"user": "my herbs keep getting leggy", "assistant": "listen"},
    {"user": "no voice", "assistant": "Add an LED grow light"},
    {"user": "no voice", "assistant": "twelve inches above the canopy."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 3 loaded: emotional_support, filler_and_mhm")

# 9. hesitation_wait (7 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00148"] = [
    {"user": "Well, I was thinking about", "assistant": "listen"},
    {"user": "maybe moving to another city", "assistant": "listen"},
    {"user": "though leaving friends feels scary", "assistant": "listen"},
    {"user": "no voice", "assistant": "Relocation brings mixed feelings."},
    {"user": "I am mainly hesitant about", "assistant": "listen"},
    {"user": "cost of living and rent", "assistant": "listen"},
    {"user": "no voice", "assistant": "We can calculate monthly expenses"},
    {"user": "no voice", "assistant": "versus your projected salary change."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00106"] = [
    {"user": "Um, I have been considering", "assistant": "listen"},
    {"user": "applying for that management position", "assistant": "listen"},
    {"user": "but I lack senior tenure", "assistant": "listen"},
    {"user": "no voice", "assistant": "That is understandable doubt."},
    {"user": "I really worry that", "assistant": "listen"},
    {"user": "people will question my leadership", "assistant": "listen"},
    {"user": "no voice", "assistant": "Focus on your project outcomes"},
    {"user": "no voice", "assistant": "and proven team mentorship history."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00463"] = [
    {"user": "Well, I am thinking of", "assistant": "listen"},
    {"user": "quitting my corporate banking job", "assistant": "listen"},
    {"user": "to teach mathematics instead", "assistant": "listen"},
    {"user": "no voice", "assistant": "A significant career transition."},
    {"user": "my biggest fear is", "assistant": "listen"},
    {"user": "taking a substantial pay cut", "assistant": "listen"},
    {"user": "no voice", "assistant": "Build a six-month safety reserve"},
    {"user": "no voice", "assistant": "before making the formal leap."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00484"] = [
    {"user": "You see, I wanted to", "assistant": "listen"},
    {"user": "adopt a high-energy rescue puppy", "assistant": "listen"},
    {"user": "but my apartment is small", "assistant": "listen"},
    {"user": "no voice", "assistant": "A thoughtful concern to have."},
    {"user": "I am concerned about", "assistant": "listen"},
    {"user": "leaving him while at work", "assistant": "listen"},
    {"user": "no voice", "assistant": "Consider daily doggy daycare services"},
    {"user": "no voice", "assistant": "or a midday professional walker."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00085"] = [
    {"user": "So, I have been debating", "assistant": "listen"},
    {"user": "buying an electric commuter car", "assistant": "listen"},
    {"user": "versus keeping my hybrid vehicle", "assistant": "listen"},
    {"user": "no voice", "assistant": "Both have solid benefits."},
    {"user": "I am really unsure whether", "assistant": "listen"},
    {"user": "home charger installation makes sense", "assistant": "listen"},
    {"user": "no voice", "assistant": "Check local electrical panel capacity"},
    {"user": "no voice", "assistant": "and utility off-peak rebate programs."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00526"] = [
    {"user": "Well, I am trying to", "assistant": "listen"},
    {"user": "decide on graduate school programs", "assistant": "listen"},
    {"user": "either data science or robotics", "assistant": "listen"},
    {"user": "no voice", "assistant": "Both offer strong career pathways."},
    {"user": "my main reservation is", "assistant": "listen"},
    {"user": "taking on student loan debt", "assistant": "listen"},
    {"user": "no voice", "assistant": "Look for fully funded assistantships"},
    {"user": "no voice", "assistant": "that waive tuition and stipends."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00316"] = [
    {"user": "Um, I was hoping to", "assistant": "listen"},
    {"user": "start selling ceramic mugs online", "assistant": "listen"},
    {"user": "but pricing feels so arbitrary", "assistant": "listen"},
    {"user": "no voice", "assistant": "Pricing handmade art takes thought."},
    {"user": "I get anxious about", "assistant": "listen"},
    {"user": "charging enough for my hours", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sum material costs and studio overhead"},
    {"user": "no voice", "assistant": "plus your target hourly wage."},
    {"user": "no voice", "assistant": "listen"},
]

# 10. long_explanation_user (10 items):
# Template length: 12 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:listen
# 3: U:SPEAK | A:listen
# 4: U:SPEAK | A:listen
# 5: U:no voice | A:SPEAK
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:SPEAK
# 10: U:no voice | A:SPEAK
# 11: U:no voice | A:listen

CONVERSATIONS["conv_00349"] = [
    {"user": "I am trying to learn conversational Italian", "assistant": "listen"},
    {"user": "before visiting Florence this summer", "assistant": "listen"},
    {"user": "I downloaded several mobile vocabulary flashcards", "assistant": "listen"},
    {"user": "and listen to language podcasts", "assistant": "listen"},
    {"user": "but I freeze when speaking aloud", "assistant": "listen"},
    {"user": "no voice", "assistant": "Speaking hesitation is completely natural."},
    {"user": "maybe find an online tandem partner", "assistant": "listen"},
    {"user": "no voice", "assistant": "Practice everyday greetings out loud,"},
    {"user": "no voice", "assistant": "record your voice reading menus,"},
    {"user": "no voice", "assistant": "and practice brief cafe roleplays."},
    {"user": "no voice", "assistant": "Consistency matters more than perfection."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00034"] = [
    {"user": "I started training for a marathon", "assistant": "listen"},
    {"user": "running four mornings every week", "assistant": "listen"},
    {"user": "my cardiovascular endurance feels quite strong", "assistant": "listen"},
    {"user": "but my right knee aches badly", "assistant": "listen"},
    {"user": "after passing the eight-mile threshold", "assistant": "listen"},
    {"user": "no voice", "assistant": "Joint pain warrants careful attention."},
    {"user": "should I replace my running sneakers", "assistant": "listen"},
    {"user": "no voice", "assistant": "Get your gait analyzed locally,"},
    {"user": "no voice", "assistant": "strengthen your hips and glutes,"},
    {"user": "no voice", "assistant": "and lower mileage temporarily."},
    {"user": "no voice", "assistant": "Rest allows irritated tendons healing."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00517"] = [
    {"user": "I am setting up a home vegetable garden", "assistant": "listen"},
    {"user": "with three raised wooden planter beds", "assistant": "listen"},
    {"user": "the yard has rich sandy loam soil", "assistant": "listen"},
    {"user": "and gets full southern sun exposure", "assistant": "listen"},
    {"user": "but local deer jump the fence", "assistant": "listen"},
    {"user": "no voice", "assistant": "Wildlife fencing requires proper height."},
    {"user": "maybe install netting above the beds", "assistant": "listen"},
    {"user": "no voice", "assistant": "Erect eight-foot perimeter deer netting,"},
    {"user": "no voice", "assistant": "or enclose individual raised beds"},
    {"user": "no voice", "assistant": "with sturdy hinged wire cages."},
    {"user": "no voice", "assistant": "That reliably preserves tender shoots."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00286"] = [
    {"user": "I want to organize digital clutter", "assistant": "listen"},
    {"user": "across three external hard drives", "assistant": "listen"},
    {"user": "I have ten years of photographs", "assistant": "listen"},
    {"user": "mixed with college tax documents", "assistant": "listen"},
    {"user": "and thousands of duplicate downloads", "assistant": "listen"},
    {"user": "no voice", "assistant": "A methodical system reduces overwhelm."},
    {"user": "how should I structure folders", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sort primary archives by year,"},
    {"user": "no voice", "assistant": "run an automated deduplication scan,"},
    {"user": "no voice", "assistant": "and mirror to encrypted cloud."},
    {"user": "no voice", "assistant": "Tackle one folder batch daily."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00538"] = [
    {"user": "I manage remote engineering team standups", "assistant": "listen"},
    {"user": "across four global time zones", "assistant": "listen"},
    {"user": "people join tired from late hours", "assistant": "listen"},
    {"user": "discussions drag past thirty minutes", "assistant": "listen"},
    {"user": "and blockers rarely get resolved", "assistant": "listen"},
    {"user": "no voice", "assistant": "Asynchronous updates often work better."},
    {"user": "should we switch to Slack checkins", "assistant": "listen"},
    {"user": "no voice", "assistant": "Adopt written daily channel updates,"},
    {"user": "no voice", "assistant": "reserve video for thorny blockers,"},
    {"user": "no voice", "assistant": "and keep calls under fifteen."},
    {"user": "no voice", "assistant": "Your distributed team will appreciate."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00307"] = [
    {"user": "I am renovating my old kitchen", "assistant": "listen"},
    {"user": "on a fairly strict budget", "assistant": "listen"},
    {"user": "the wooden cabinet boxes are solid", "assistant": "listen"},
    {"user": "but the laminate countertops look peeling", "assistant": "listen"},
    {"user": "and drawer fronts are dated", "assistant": "listen"},
    {"user": "no voice", "assistant": "Preserving core cabinet boxes saves money."},
    {"user": "can I just paint and reface", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sand and repaint existing boxes,"},
    {"user": "no voice", "assistant": "swap out modern brushed hardware,"},
    {"user": "no voice", "assistant": "and install durable butcher block."},
    {"user": "no voice", "assistant": "That creates a fresh look."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00097"] = [
    {"user": "I started writing a fantasy novel", "assistant": "listen"},
    {"user": "outlining three complex magical factions", "assistant": "listen"},
    {"user": "drafted fifty pages of world history", "assistant": "listen"},
    {"user": "and designed intricate geography maps", "assistant": "listen"},
    {"user": "but chapter one has no plot", "assistant": "listen"},
    {"user": "no voice", "assistant": "Worldbuilding without character tension stalls."},
    {"user": "how do I kickstart the conflict", "assistant": "listen"},
    {"user": "no voice", "assistant": "Give your protagonist immediate stakes,"},
    {"user": "no voice", "assistant": "introduce a tangible urgent problem,"},
    {"user": "no voice", "assistant": "and reveal lore through action."},
    {"user": "no voice", "assistant": "Plot springs from personal desire."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00496"] = [
    {"user": "I am moving into my van", "assistant": "listen"},
    {"user": "to travel national parks full-time", "assistant": "listen"},
    {"user": "I installed solar roof panels", "assistant": "listen"},
    {"user": "and built an insulated cedar bed", "assistant": "listen"},
    {"user": "but off-grid refrigeration drains power", "assistant": "listen"},
    {"user": "no voice", "assistant": "Battery balancing requires efficient appliances."},
    {"user": "should I add another lithium battery", "assistant": "listen"},
    {"user": "no voice", "assistant": "Upgrade to high-efficiency twelve-volt cooling,"},
    {"user": "no voice", "assistant": "add an alternator DC charger,"},
    {"user": "no voice", "assistant": "and expand battery bank capacity."},
    {"user": "no voice", "assistant": "That ensures stable off-grid power."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00244"] = [
    {"user": "I brew kombucha at home", "assistant": "listen"},
    {"user": "using organic sweet black tea", "assistant": "listen"},
    {"user": "first fermentation goes great every time", "assistant": "listen"},
    {"user": "yielding crisp tart fermented brew", "assistant": "listen"},
    {"user": "but second bottling produces no carbonation", "assistant": "listen"},
    {"user": "no voice", "assistant": "Carbonation depends on airtight sealing."},
    {"user": "what creates tight bubbly fizz", "assistant": "listen"},
    {"user": "no voice", "assistant": "Use swing-top pressure rated bottles,"},
    {"user": "no voice", "assistant": "add fresh fruit ginger puree,"},
    {"user": "no voice", "assistant": "and keep at warm temperatures."},
    {"user": "no voice", "assistant": "Residual yeast needs sugar warmth."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00433"] = [
    {"user": "I want to launch a podcast", "assistant": "listen"},
    {"user": "interviewing local artisan coffee roasters", "assistant": "listen"},
    {"user": "I bought dynamic XLR microphones", "assistant": "listen"},
    {"user": "and lined a closet with foam", "assistant": "listen"},
    {"user": "but background room echo still persists", "assistant": "listen"},
    {"user": "no voice", "assistant": "Hard surfaces bounce acoustic reverberation."},
    {"user": "how can I dampen the room", "assistant": "listen"},
    {"user": "no voice", "assistant": "Hang heavy moving blankets around,"},
    {"user": "no voice", "assistant": "place thick rugs on floor,"},
    {"user": "no voice", "assistant": "and keep mouths close on-mic."},
    {"user": "no voice", "assistant": "Proximity reduces room noise bleed."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 4 loaded: hesitation_wait, long_explanation_user")

# 11. misheard_word (5 items):
# Template length: 8 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:no voice | A:SPEAK
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:listen

CONVERSATIONS["conv_00544"] = [
    {"user": "Find a good recipe for beet soup", "assistant": "listen"},
    {"user": "no voice", "assistant": "Did you mean red beet soup"},
    {"user": "no voice", "assistant": "or savory beef stew soup?"},
    {"user": "red beets, like traditional Ukrainian borscht", "assistant": "listen"},
    {"user": "no voice", "assistant": "Got it: red beet borscht."},
    {"user": "no voice", "assistant": "Simmer shredded beets with potatoes,"},
    {"user": "no voice", "assistant": "carrots, and a touch of vinegar."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00565"] = [
    {"user": "Check flight prices to Austin", "assistant": "listen"},
    {"user": "no voice", "assistant": "Are you flying to Austin Texas"},
    {"user": "no voice", "assistant": "or Boston Logan International Airport?"},
    {"user": "Austin Texas, live music festival weekend", "assistant": "listen"},
    {"user": "no voice", "assistant": "Understood: Austin Bergstrom International Airport."},
    {"user": "no voice", "assistant": "Nonstop flights begin around two-fifty."},
    {"user": "no voice", "assistant": "Would you like morning departures?"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00229"] = [
    {"user": "How do I care for my hare", "assistant": "listen"},
    {"user": "no voice", "assistant": "Are you asking about curly hair"},
    {"user": "no voice", "assistant": "or a pet pet hare rabbit?"},
    {"user": "pet domestic rabbit, my speech garbled", "assistant": "listen"},
    {"user": "no voice", "assistant": "Noted: pet rabbit care advice."},
    {"user": "no voice", "assistant": "Provide unlimited timothy grass hay,"},
    {"user": "no voice", "assistant": "fresh water, and leafy greens."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00523"] = [
    {"user": "Where can I purchase a crate", "assistant": "listen"},
    {"user": "no voice", "assistant": "Do you need a wooden shipping crate"},
    {"user": "no voice", "assistant": "or a folding dog training crate?"},
    {"user": "a wire dog crate for my puppy", "assistant": "listen"},
    {"user": "no voice", "assistant": "Got it: wire puppy training crate."},
    {"user": "no voice", "assistant": "Pet supply stores offer folding models"},
    {"user": "no voice", "assistant": "with adjustable internal divider panels."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00481"] = [
    {"user": "Tell me about the knight", "assistant": "listen"},
    {"user": "no voice", "assistant": "Do you mean medieval armored knights"},
    {"user": "no voice", "assistant": "or astronomical starry night sky?"},
    {"user": "medieval knights and their chivalric code", "assistant": "listen"},
    {"user": "no voice", "assistant": "Understood: medieval European chivalry history."},
    {"user": "no voice", "assistant": "Knights followed aristocratic martial customs,"},
    {"user": "no voice", "assistant": "combining combat valor with etiquette."},
    {"user": "no voice", "assistant": "listen"},
]

# 12. multistep_task (4 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00438"] = [
    {"user": "Help me declutter my closet", "assistant": "listen"},
    {"user": "before moving next month", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sort items into three distinct piles."},
    {"user": "keep, donate, and discard", "assistant": "listen"},
    {"user": "no voice", "assistant": "Exactly. Try on questionable clothing items."},
    {"user": "and pack the keeps into boxes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Label wardrobe boxes clearly by season"},
    {"user": "no voice", "assistant": "and drop donations off promptly."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00291"] = [
    {"user": "Guide me through winterizing pipes", "assistant": "listen"},
    {"user": "in an unheated garden shed", "assistant": "listen"},
    {"user": "no voice", "assistant": "Shut off the interior supply valve."},
    {"user": "open external spigots to drain", "assistant": "listen"},
    {"user": "no voice", "assistant": "Let remaining water drain out fully."},
    {"user": "and wrap pipes with foam", "assistant": "listen"},
    {"user": "no voice", "assistant": "Fit pre-slit foam insulation jackets"},
    {"user": "no voice", "assistant": "and secure joints with tape."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00102"] = [
    {"user": "How do I make strawberry jam", "assistant": "listen"},
    {"user": "from freshly picked berries", "assistant": "listen"},
    {"user": "no voice", "assistant": "Hull and crush berries with sugar."},
    {"user": "bring the fruit to boil", "assistant": "listen"},
    {"user": "no voice", "assistant": "Boil vigorously until setting point."},
    {"user": "then ladle into sanitized jars", "assistant": "listen"},
    {"user": "no voice", "assistant": "Wipe jar rims completely clean"},
    {"user": "no voice", "assistant": "and process in boiling water bath."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00501"] = [
    {"user": "Help me prepare my car", "assistant": "listen"},
    {"user": "for a cross-country summer trip", "assistant": "listen"},
    {"user": "no voice", "assistant": "Inspect all tire pressures and tread."},
    {"user": "check engine oil and coolant", "assistant": "listen"},
    {"user": "no voice", "assistant": "Top up clean motor fluids properly."},
    {"user": "and pack an emergency roadside kit", "assistant": "listen"},
    {"user": "no voice", "assistant": "Include jumper cables, flashlight, tools,"},
    {"user": "no voice", "assistant": "and extra drinking water jugs."},
    {"user": "no voice", "assistant": "listen"},
]

# 13. natural_conversation (2 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00231"] = [
    {"user": "I tried roasting almonds today", "assistant": "listen"},
    {"user": "but they came out burnt", "assistant": "listen"},
    {"user": "no voice", "assistant": "Nuts burn very quickly unattended."},
    {"user": "I left the oven at four hundred", "assistant": "listen"},
    {"user": "no voice", "assistant": "Try three hundred twenty-five degrees"},
    {"user": "no voice", "assistant": "and toss almonds every five minutes."},
    {"user": "I will watch them closer next time", "assistant": "listen"},
    {"user": "no voice", "assistant": "They will toast wonderfully even."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00126"] = [
    {"user": "My houseplant leaves are yellowing", "assistant": "listen"},
    {"user": "even though I water daily", "assistant": "listen"},
    {"user": "no voice", "assistant": "Daily watering often causes root rot."},
    {"user": "the potting soil stays quite soggy", "assistant": "listen"},
    {"user": "no voice", "assistant": "Let the top two inches dry"},
    {"user": "no voice", "assistant": "and ensure pot drainage holes open."},
    {"user": "I will ease up on the watering can", "assistant": "listen"},
    {"user": "no voice", "assistant": "Your plant will bounce back nicely."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 5 loaded: misheard_word, multistep_task, natural_conversation")

# 14. question_answer_chunks (9 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:listen
# 3: U:no voice | A:SPEAK
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00191"] = [
    {"user": "How does photosynthesis work", "assistant": "listen"},
    {"user": "inside green plant leaves", "assistant": "listen"},
    {"user": "explained simply for kids", "assistant": "listen"},
    {"user": "no voice", "assistant": "Chlorophyll absorbs radiant sun rays."},
    {"user": "no voice", "assistant": "Leaves take carbon dioxide from air"},
    {"user": "no voice", "assistant": "and roots absorb ground water."},
    {"user": "no voice", "assistant": "They produce energy-rich sugars"},
    {"user": "no voice", "assistant": "and release pure oxygen gas."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00023"] = [
    {"user": "Why do ocean tides happen", "assistant": "listen"},
    {"user": "rising and falling twice daily", "assistant": "listen"},
    {"user": "in basic physical terms", "assistant": "listen"},
    {"user": "no voice", "assistant": "Moon gravity pulls on Earth oceans."},
    {"user": "no voice", "assistant": "Water bulges on opposite sides,"},
    {"user": "no voice", "assistant": "creating high tides twice daily."},
    {"user": "no voice", "assistant": "As Earth rotates beneath bulges,"},
    {"user": "no voice", "assistant": "shorelines experience cyclical tides."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00275"] = [
    {"user": "What causes thunderstorms to form", "assistant": "listen"},
    {"user": "on hot humid summer afternoons", "assistant": "listen"},
    {"user": "can you explain simply", "assistant": "listen"},
    {"user": "no voice", "assistant": "Warm humid air rises rapidly."},
    {"user": "no voice", "assistant": "Moisture condenses into towering clouds,"},
    {"user": "no voice", "assistant": "releasing powerful latent thermal heat."},
    {"user": "no voice", "assistant": "Ice particles collide generating electricity,"},
    {"user": "no voice", "assistant": "sparking lightning and loud thunder."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00044"] = [
    {"user": "How does noise cancellation work", "assistant": "listen"},
    {"user": "inside modern wireless headphones", "assistant": "listen"},
    {"user": "without muting the music", "assistant": "listen"},
    {"user": "no voice", "assistant": "External microphones sample ambient noise."},
    {"user": "no voice", "assistant": "Internal chips synthesize inverted waves."},
    {"user": "no voice", "assistant": "The inverse wave cancels noise,"},
    {"user": "no voice", "assistant": "leaving clean audio playback untouched."},
    {"user": "no voice", "assistant": "That creates silent listening environments."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00317"] = [
    {"user": "Why do honeybees make honey", "assistant": "listen"},
    {"user": "inside their hive colonies", "assistant": "listen"},
    {"user": "during spring and summer", "assistant": "listen"},
    {"user": "no voice", "assistant": "Bees collect flower nectar drops."},
    {"user": "no voice", "assistant": "They evaporate water by fanning,"},
    {"user": "no voice", "assistant": "concentrating rich sugary carbohydrate reserves."},
    {"user": "no voice", "assistant": "The honey feeds the hive"},
    {"user": "no voice", "assistant": "through freezing cold winter months."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00128"] = [
    {"user": "How do solar panels generate", "assistant": "listen"},
    {"user": "usable electricity from sunlight", "assistant": "listen"},
    {"user": "on residential house roofs", "assistant": "listen"},
    {"user": "no voice", "assistant": "Photons strike silicon semiconductor cells."},
    {"user": "no voice", "assistant": "Dislodged electrons flow through circuit,"},
    {"user": "no voice", "assistant": "producing direct electrical current power."},
    {"user": "no voice", "assistant": "An inverter transforms that DC"},
    {"user": "no voice", "assistant": "into household alternating current electricity."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00359"] = [
    {"user": "Why do leaves change color", "assistant": "listen"},
    {"user": "every autumn before falling", "assistant": "listen"},
    {"user": "in deciduous forest trees", "assistant": "listen"},
    {"user": "no voice", "assistant": "Shorter days reduce green chlorophyll."},
    {"user": "no voice", "assistant": "Underlying orange carotenoid pigments emerge."},
    {"user": "no voice", "assistant": "Trapped sugars produce crimson anthocyanins,"},
    {"user": "no voice", "assistant": "turning foliage brilliant red hues."},
    {"user": "no voice", "assistant": "Trees shed leaves before winter."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00296"] = [
    {"user": "How do microwave ovens heat", "assistant": "listen"},
    {"user": "cold food so quickly", "assistant": "listen"},
    {"user": "compared to regular ovens", "assistant": "listen"},
    {"user": "no voice", "assistant": "Magnetrons emit high frequency microwaves."},
    {"user": "no voice", "assistant": "Waves excite internal water molecules,"},
    {"user": "no voice", "assistant": "causing them to vibrate rapidly."},
    {"user": "no voice", "assistant": "Molecular friction generates internal heat,"},
    {"user": "no voice", "assistant": "warming food evenly from inside."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00002"] = [
    {"user": "Why does ice float on", "assistant": "listen"},
    {"user": "liquid water in glasses", "assistant": "listen"},
    {"user": "unlike most other liquids", "assistant": "listen"},
    {"user": "no voice", "assistant": "Water molecules form crystalline lattices."},
    {"user": "no voice", "assistant": "Hydrogen bonds expand upon freezing,"},
    {"user": "no voice", "assistant": "trapping empty microscopic space between."},
    {"user": "no voice", "assistant": "This makes solid ice less dense"},
    {"user": "no voice", "assistant": "than surrounding liquid water molecules."},
    {"user": "no voice", "assistant": "listen"},
]

# 15. short_acknowledgement (4 items):
# Template length: 8 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:listen

CONVERSATIONS["conv_00406"] = [
    {"user": "I passed my road driving test", "assistant": "listen"},
    {"user": "on the first attempt today", "assistant": "listen"},
    {"user": "no voice", "assistant": "Congratulations on the pass."},
    {"user": "I parallel parked without touching cones", "assistant": "listen"},
    {"user": "no voice", "assistant": "That takes solid car control."},
    {"user": "time to drive independently now", "assistant": "listen"},
    {"user": "no voice", "assistant": "Drive safely out there today."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00154"] = [
    {"user": "I submitted my tax return", "assistant": "listen"},
    {"user": "well before the April deadline", "assistant": "listen"},
    {"user": "no voice", "assistant": "Great filing efficiency there."},
    {"user": "it took all weekend gathering receipts", "assistant": "listen"},
    {"user": "no voice", "assistant": "Tax paperwork is quite tedious."},
    {"user": "glad that chore is finished", "assistant": "listen"},
    {"user": "no voice", "assistant": "Enjoy your stress-free weekend."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00049"] = [
    {"user": "I assembled the nursery crib", "assistant": "listen"},
    {"user": "without missing any hardware screws", "assistant": "listen"},
    {"user": "no voice", "assistant": "Well done assembling it."},
    {"user": "the instruction booklet was confusing", "assistant": "listen"},
    {"user": "no voice", "assistant": "Flat-pack furniture diagrams test patience."},
    {"user": "now the room looks ready", "assistant": "listen"},
    {"user": "no voice", "assistant": "Such an exciting life milestone."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00133"] = [
    {"user": "I ran my first half-marathon", "assistant": "listen"},
    {"user": "and crossed under two hours", "assistant": "listen"},
    {"user": "no voice", "assistant": "Incredible running accomplishment there."},
    {"user": "my legs feel like solid stone", "assistant": "listen"},
    {"user": "no voice", "assistant": "Hydrate and stretch your calves."},
    {"user": "I am resting all afternoon", "assistant": "listen"},
    {"user": "no voice", "assistant": "Well-deserved recovery rest today."},
    {"user": "no voice", "assistant": "listen"},
]

# 16. silence_then_resume (8 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:listen  <-- note assistant stays listen!
# 3: U:SPEAK | A:listen
# 4: U:SPEAK | A:listen
# 5: U:no voice | A:SPEAK
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00199"] = [
    {"user": "I heard something intriguing about", "assistant": "listen"},
    {"user": "deep sea bioluminescent squid today", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "they use glowing light organs", "assistant": "listen"},
    {"user": "to camouflage against ocean sunlight", "assistant": "listen"},
    {"user": "no voice", "assistant": "Counterillumination is fascinating survival biology."},
    {"user": "it hides their silhouettes completely", "assistant": "listen"},
    {"user": "no voice", "assistant": "Predators below see only light."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00073"] = [
    {"user": "I was reading an article", "assistant": "listen"},
    {"user": "about ancient Roman concrete architecture", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "they mixed volcanic ash pozzolana", "assistant": "listen"},
    {"user": "which actually self-heals salt cracks", "assistant": "listen"},
    {"user": "no voice", "assistant": "Hot-mixing lime created reactive clasts."},
    {"user": "that explains maritime harbour durability", "assistant": "listen"},
    {"user": "no voice", "assistant": "Structures survived two thousand years."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00094"] = [
    {"user": "I discovered an interesting fact", "assistant": "listen"},
    {"user": "about monarch butterfly migration routes", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "they navigate across three countries", "assistant": "listen"},
    {"user": "using magnetic antennae sun compasses", "assistant": "listen"},
    {"user": "no voice", "assistant": "Multigenerational navigation is astonishingly complex."},
    {"user": "the fourth generation flies back", "assistant": "listen"},
    {"user": "no voice", "assistant": "They find specific overwintering firs."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00178"] = [
    {"user": "I came across some research", "assistant": "listen"},
    {"user": "on how mycelium fungal networks", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "connect different tree root systems", "assistant": "listen"},
    {"user": "distributing carbon and nitrogen nutrients", "assistant": "listen"},
    {"user": "no voice", "assistant": "Forest mycorrhizal networks act cooperatively."},
    {"user": "mother trees nurture smaller seedlings", "assistant": "listen"},
    {"user": "no voice", "assistant": "Ecologists call it wood-wide web."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00157"] = [
    {"user": "I was checking space news", "assistant": "listen"},
    {"user": "about the James Webb telescope", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "it detected carbon dioxide atmospheres", "assistant": "listen"},
    {"user": "on a distant gas exoplanet", "assistant": "listen"},
    {"user": "no voice", "assistant": "Transmission spectroscopy provides detailed data."},
    {"user": "spectroscopy reveals chemical atmospheric fingerprints", "assistant": "listen"},
    {"user": "no voice", "assistant": "Key step toward biosignature detection."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00241"] = [
    {"user": "I watched a fascinating documentary", "assistant": "listen"},
    {"user": "on raven social cognitive intelligence", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "they solve multi-stage puzzle boxes", "assistant": "listen"},
    {"user": "and understand human gaze direction", "assistant": "listen"},
    {"user": "no voice", "assistant": "Corvid problem-solving rivals primate intelligence."},
    {"user": "they even remember friendly faces", "assistant": "listen"},
    {"user": "no voice", "assistant": "They pass facial recognition culturally."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00514"] = [
    {"user": "I was learning about acoustics", "assistant": "listen"},
    {"user": "in ancient Greek stone theaters", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "the tiered limestone seating rows", "assistant": "listen"},
    {"user": "naturally filter out low-frequency noise", "assistant": "listen"},
    {"user": "no voice", "assistant": "Epidaurus theater acoustic engineering marvel."},
    {"user": "actors voices carry across fifty rows", "assistant": "listen"},
    {"user": "no voice", "assistant": "Unamplified sound travels with clarity."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00556"] = [
    {"user": "I was reading about sleep", "assistant": "listen"},
    {"user": "in arctic migratory seabird flocks", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
    {"user": "they engage unihemispheric slow-wave sleep", "assistant": "listen"},
    {"user": "resting half their brain mid-flight", "assistant": "listen"},
    {"user": "no voice", "assistant": "One eye remains visually alert."},
    {"user": "allowing continuous non-stop transoceanic flights", "assistant": "listen"},
    {"user": "no voice", "assistant": "Remarkable physiological adaptation for migration."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 6 loaded: question_answer_chunks, short_acknowledgement, silence_then_resume")

# 17. urgent_chemical_safety (9 items):
# Template length: 10 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:SPEAK | A:SPEAK  <-- immediate warning!
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:SPEAK  <-- immediate warning!
# 5: U:SPEAK | A:SPEAK  <-- immediate warning!
# 6: U:no voice | A:SPEAK
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:listen

CONVERSATIONS["conv_00278"] = [
    {"user": "I am in my garage workshop", "assistant": "listen"},
    {"user": "I have bleach on the floor", "assistant": "listen"},
    {"user": "and pouring white vinegar directly on", "assistant": "Stop! Never mix those!"},
    {"user": "no voice", "assistant": "That creates lethal chlorine gas."},
    {"user": "a sharp yellow cloud is forming", "assistant": "Evacuate the garage immediately."},
    {"user": "my eyes are burning fiercely", "assistant": "Get into fresh air now."},
    {"user": "no voice", "assistant": "Do not breathe any fumes."},
    {"user": "no voice", "assistant": "Call emergency nine-one-one immediately"},
    {"user": "no voice", "assistant": "and keep garage doors closed."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00257"] = [
    {"user": "Cleaning grout in my kitchen", "assistant": "listen"},
    {"user": "I mixed household pool chlorine", "assistant": "listen"},
    {"user": "with concentrated muriatic acid cleaner", "assistant": "Do not mix acid chlorine!"},
    {"user": "no voice", "assistant": "It releases toxic chlorine gas."},
    {"user": "it is fizzing violently green", "assistant": "Back away from it now."},
    {"user": "my chest feels super tight", "assistant": "Leave the house immediately."},
    {"user": "no voice", "assistant": "Seek fresh air outside now."},
    {"user": "no voice", "assistant": "Call emergency medical services immediately"},
    {"user": "no voice", "assistant": "for acute toxic fume inhalation."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00467"] = [
    {"user": "I am unstopping laundry pipes", "assistant": "listen"},
    {"user": "I poured commercial lye drain cleaner", "assistant": "listen"},
    {"user": "and then added sulfuric acid", "assistant": "Stop immediately! Extreme danger!"},
    {"user": "no voice", "assistant": "Acid base reactions cause explosions."},
    {"user": "it is violently boiling over", "assistant": "Do not touch boiling liquid."},
    {"user": "boiling droplets splattered my arm", "assistant": "Rinse arm with cold water."},
    {"user": "no voice", "assistant": "Flush skin for twenty minutes."},
    {"user": "no voice", "assistant": "Seek emergency medical treatment immediately"},
    {"user": "no voice", "assistant": "for severe chemical caustic burns."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00362"] = [
    {"user": "I am disinfecting kitchen counters", "assistant": "listen"},
    {"user": "I sprayed hydrogen peroxide everywhere", "assistant": "listen"},
    {"user": "and spraying vinegar right over it", "assistant": "Stop! Do not mix them!"},
    {"user": "no voice", "assistant": "They combine into peracetic acid."},
    {"user": "the vapor is very pungent", "assistant": "Move away from kitchen counters."},
    {"user": "my throat hurts from breathing", "assistant": "Step into fresh air outside."},
    {"user": "no voice", "assistant": "Peracetic vapors burn respiratory tracts."},
    {"user": "no voice", "assistant": "Contact poison control center immediately"},
    {"user": "no voice", "assistant": "and ventilate the entire area."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00572"] = [
    {"user": "I am stripping furniture finish", "assistant": "listen"},
    {"user": "using industrial solvent paint stripper", "assistant": "listen"},
    {"user": "in a closed windowless basement", "assistant": "Open windows or stop now!"},
    {"user": "no voice", "assistant": "Solvent fumes pose severe hazards."},
    {"user": "I smell intense chemical sweetness", "assistant": "Leave that basement right now."},
    {"user": "I am feeling very dizzy", "assistant": "Climb upstairs into open air."},
    {"user": "no voice", "assistant": "Dichloromethane vapors cause sudden asphyxiation."},
    {"user": "no voice", "assistant": "Stay in open outdoor air"},
    {"user": "no voice", "assistant": "and request medical assistance if faint."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00089"] = [
    {"user": "I am sterilizing garden tools", "assistant": "listen"},
    {"user": "I filled a plastic tub with bleach", "assistant": "listen"},
    {"user": "and dumped rubbing alcohol in", "assistant": "Stop! Do not mix alcohol!"},
    {"user": "no voice", "assistant": "Bleach and alcohol create chloroform."},
    {"user": "a weird sweet smell arose", "assistant": "Step back from that tub."},
    {"user": "I feel drowsy and disoriented", "assistant": "Exit the shed right now."},
    {"user": "no voice", "assistant": "Chloroform gas suppresses the nervous system."},
    {"user": "no voice", "assistant": "Call local emergency services immediately"},
    {"user": "no voice", "assistant": "and keep people well clear."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00446"] = [
    {"user": "I am cleaning an oven", "assistant": "listen"},
    {"user": "I sprayed heavy chemical oven foam", "assistant": "listen"},
    {"user": "then turned oven heat on full", "assistant": "Turn off heat! Severe danger!"},
    {"user": "no voice", "assistant": "Heated lye fumes are corrosive."},
    {"user": "white smoke is pouring out", "assistant": "Shut oven door and retreat."},
    {"user": "the smoke is suffocating me", "assistant": "Evacuate the house immediately now."},
    {"user": "no voice", "assistant": "Avoid inhaling caustic oven fumes."},
    {"user": "no voice", "assistant": "Dial nine-one-one for emergency help"},
    {"user": "no voice", "assistant": "and do not re-enter premises."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00341"] = [
    {"user": "Doing yard weed maintenance", "assistant": "listen"},
    {"user": "I found an old pesticide bottle", "assistant": "listen"},
    {"user": "the glass bottle cracked open", "assistant": "Avoid skin contact! Back up!"},
    {"user": "no voice", "assistant": "Concentrated pesticides absorb through skin."},
    {"user": "it spilled over my bare feet", "assistant": "Strip shoes and flush feet."},
    {"user": "my skin is stinging intensely", "assistant": "Wash with copious cold water."},
    {"user": "no voice", "assistant": "Scrub thoroughly with mild soap."},
    {"user": "no voice", "assistant": "Call emergency poison control hotline"},
    {"user": "no voice", "assistant": "with pesticide container chemical details."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00299"] = [
    {"user": "I am cleaning automotive parts", "assistant": "listen"},
    {"user": "with pressurized brake fluid spray", "assistant": "listen"},
    {"user": "next to an open blowtorch flame", "assistant": "Extinguish torch flame immediately!"},
    {"user": "no voice", "assistant": "Aerosol solvents are extremely explosive."},
    {"user": "the spray cloud touched flame", "assistant": "Get away from the workbench!"},
    {"user": "a fire ball just flared", "assistant": "Grab fire extinguisher or run."},
    {"user": "no voice", "assistant": "Do not throw water on chemical fire."},
    {"user": "no voice", "assistant": "Call the fire department immediately"},
    {"user": "no voice", "assistant": "and clear everyone outside safely."},
    {"user": "no voice", "assistant": "listen"},
]

# 18. urgent_electrical_safety (10 items):
# Template length: 7 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:SPEAK  <-- immediate warning!
# 2: U:SPEAK | A:SPEAK  <-- immediate warning!
# 3: U:no voice | A:SPEAK
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:no voice | A:listen

CONVERSATIONS["conv_00153"] = [
    {"user": "The kitchen toaster is smoking", "assistant": "listen"},
    {"user": "I will stick a metal knife inside", "assistant": "Do not insert any knife!"},
    {"user": "to pry the jammed toast loose", "assistant": "Metal conducts lethal high voltage."},
    {"user": "no voice", "assistant": "Unplug appliance from wall socket."},
    {"user": "no voice", "assistant": "Never touch interior electrical elements."},
    {"user": "no voice", "assistant": "Wait until cold before inspecting."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00321"] = [
    {"user": "A bathroom hairdryer fell in the tub", "assistant": "listen"},
    {"user": "I am reaching my hand into the water", "assistant": "Do not touch the water!"},
    {"user": "to pull the cord out quick", "assistant": "Stay away from wet tub."},
    {"user": "no voice", "assistant": "Cut bathroom breaker switch immediately."},
    {"user": "no voice", "assistant": "Water conducts lethal electrical current."},
    {"user": "no voice", "assistant": "Only unplug after power isolated."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00258"] = [
    {"user": "Storm winds blew a power line down", "assistant": "listen"},
    {"user": "I am walking toward the wire", "assistant": "Stay back! Extremely dangerous!"},
    {"user": "to move it off my driveway", "assistant": "Do not approach that wire."},
    {"user": "no voice", "assistant": "Keep at least thirty-five feet back."},
    {"user": "no voice", "assistant": "Downed lines electrify surrounding ground."},
    {"user": "no voice", "assistant": "Call electric utility and nine-one-one immediately."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00237"] = [
    {"user": "My lawnmower cut its extension cord", "assistant": "listen"},
    {"user": "I will tape the exposed copper", "assistant": "Do not touch bare wires!"},
    {"user": "while the mower is plugged in", "assistant": "Unplug cord from wall first."},
    {"user": "no voice", "assistant": "Live copper causes severe shocks."},
    {"user": "no voice", "assistant": "Discard sliced cords entirely today."},
    {"user": "no voice", "assistant": "Never patch outdoor extension cords."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00489"] = [
    {"user": "My phone charger is sparking violently", "assistant": "listen"},
    {"user": "I am grabbing the melted plastic", "assistant": "Do not touch that charger!"},
    {"user": "to yank it out now", "assistant": "Hands off sparking electrical plugs."},
    {"user": "no voice", "assistant": "Switch off wall outlet breaker."},
    {"user": "no voice", "assistant": "Use dry wood stick if urgent."},
    {"user": "no voice", "assistant": "Inspect outlet for electrical scorching."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00069"] = [
    {"user": "Basement water reached the wall outlets", "assistant": "listen"},
    {"user": "I am stepping into the puddle", "assistant": "Halt! Do not step in!"},
    {"user": "to check the sump pump", "assistant": "Flooded water may be energized."},
    {"user": "no voice", "assistant": "Turn off main service panel."},
    {"user": "no voice", "assistant": "Never walk into standing flooded water."},
    {"user": "no voice", "assistant": "Call emergency electrician and utility."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00363"] = [
    {"user": "The electric stove burner is crackling", "assistant": "listen"},
    {"user": "I am wiping sparks with a wet towel", "assistant": "Do not use wet towels!"},
    {"user": "to smother the little flame", "assistant": "Water conducts lethal stove current."},
    {"user": "no voice", "assistant": "Turn off stove control dials."},
    {"user": "no voice", "assistant": "Cut power at home breaker."},
    {"user": "no voice", "assistant": "Use baking soda for grease flames."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00342"] = [
    {"user": "Drilling into drywall I hit a pipe", "assistant": "listen"},
    {"user": "sparks flew and drill bit shocked me", "assistant": "Drop drill! Back away immediately!"},
    {"user": "I am pulling the drill out by hand", "assistant": "Do not touch metal drill."},
    {"user": "no voice", "assistant": "You penetrated live electrical wiring."},
    {"user": "no voice", "assistant": "Shut down main panel breaker."},
    {"user": "no voice", "assistant": "Call a licensed electrician promptly."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00468"] = [
    {"user": "A buzzing transformer box in my yard", "assistant": "listen"},
    {"user": "has its green metal door unlatched", "assistant": "Do not touch that cabinet!"},
    {"user": "I am looking inside at bus bars", "assistant": "Step away from open transformer."},
    {"user": "no voice", "assistant": "High-voltage equipment can arc through air."},
    {"user": "no voice", "assistant": "Keep children and pets away."},
    {"user": "no voice", "assistant": "Report damaged box to electric utility."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00027"] = [
    {"user": "My electric space heater cord is glowing", "assistant": "listen"},
    {"user": "I am grabbing the glowing cord", "assistant": "Do not grab that cord!"},
    {"user": "to pull it from the socket", "assistant": "Glowing cords cause severe burns."},
    {"user": "no voice", "assistant": "Switch off wall power breaker."},
    {"user": "no voice", "assistant": "Keep flammable blankets far away."},
    {"user": "no voice", "assistant": "Discard the damaged space heater."},
    {"user": "no voice", "assistant": "listen"},
]

print("Batch 7 loaded: urgent_chemical_safety, urgent_electrical_safety")

# 19. user_changes_topic (8 items):
# Template length: 11 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:SPEAK | A:listen
# 2: U:no voice | A:SPEAK
# 3: U:no voice | A:SPEAK
# 4: U:SPEAK | A:listen
# 5: U:SPEAK | A:listen
# 6: U:no voice | A:SPEAK
# 7: U:SPEAK | A:listen
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:SPEAK
# 10: U:no voice | A:listen

CONVERSATIONS["conv_00134"] = [
    {"user": "Can you recommend a podcast", "assistant": "listen"},
    {"user": "about medieval European history", "assistant": "listen"},
    {"user": "no voice", "assistant": "Listen to Tides of History"},
    {"user": "no voice", "assistant": "or Dan Carlin Hardcore History."},
    {"user": "actually, can you help with sourdough", "assistant": "listen"},
    {"user": "my loaf came out dense", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure. How long was bulk fermentation?"},
    {"user": "only about two hours at room temp", "assistant": "listen"},
    {"user": "no voice", "assistant": "Extend bulk fermentation to four hours"},
    {"user": "no voice", "assistant": "until dough rises fifty percent."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00155"] = [
    {"user": "What is the capital of Australia", "assistant": "listen"},
    {"user": "and what is its population", "assistant": "listen"},
    {"user": "no voice", "assistant": "The federal capital is Canberra,"},
    {"user": "no voice", "assistant": "with about four hundred thousand."},
    {"user": "wait, can we plan my roadtrip", "assistant": "listen"},
    {"user": "driving through Utah national parks", "assistant": "listen"},
    {"user": "no voice", "assistant": "Certainly. Which parks are top priority?"},
    {"user": "Zion and Bryce Canyon hiking trails", "assistant": "listen"},
    {"user": "no voice", "assistant": "Book Zion shuttle permits early"},
    {"user": "no voice", "assistant": "and hike Bryce Canyon rim sunrise."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00596"] = [
    {"user": "How do I remove coffee stains", "assistant": "listen"},
    {"user": "from a white cotton shirt", "assistant": "listen"},
    {"user": "no voice", "assistant": "Blot with cold soapy water,"},
    {"user": "no voice", "assistant": "then apply diluted white vinegar."},
    {"user": "never mind, help me fix code", "assistant": "listen"},
    {"user": "my Python script throws IndexError", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure. Which line triggered the error?"},
    {"user": "accessing an empty dictionary key list", "assistant": "listen"},
    {"user": "no voice", "assistant": "Check list length before indexing"},
    {"user": "no voice", "assistant": "or use dictionary get with fallback."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00344"] = [
    {"user": "Suggest a good book", "assistant": "listen"},
    {"user": "about behavioral economics research", "assistant": "listen"},
    {"user": "no voice", "assistant": "Read Thinking Fast and Slow"},
    {"user": "no voice", "assistant": "or Nudge by Richard Thaler."},
    {"user": "actually, tell me about oil changes", "assistant": "listen"},
    {"user": "how often for synthetic motor oil", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure. What vehicle make and year?"},
    {"user": "a 2020 Honda Civic commuter car", "assistant": "listen"},
    {"user": "no voice", "assistant": "Change every seven to ten thousand"},
    {"user": "no voice", "assistant": "or follow the maintenance minder indicator."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00071"] = [
    {"user": "What causes northern lights auroras", "assistant": "listen"},
    {"user": "in polar atmospheric skies", "assistant": "listen"},
    {"user": "no voice", "assistant": "Solar wind charged particles collide"},
    {"user": "no voice", "assistant": "with high altitude gas molecules."},
    {"user": "hold on, help me with gardening", "assistant": "listen"},
    {"user": "my basil leaves are wilting", "assistant": "listen"},
    {"user": "no voice", "assistant": "Of course. Is the soil dry?"},
    {"user": "very dry, soil pulling from edges", "assistant": "listen"},
    {"user": "no voice", "assistant": "Give it a thorough bottom watering"},
    {"user": "no voice", "assistant": "and keep out of direct midday scorching."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00323"] = [
    {"user": "Explain what a black hole is", "assistant": "listen"},
    {"user": "in simple non-mathematical terms", "assistant": "listen"},
    {"user": "no voice", "assistant": "A massive collapsed astronomical object"},
    {"user": "no voice", "assistant": "where gravity traps all light."},
    {"user": "actually, what should I pack", "assistant": "listen"},
    {"user": "for a rainy weekend in Seattle", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure. What activities do you have planned?"},
    {"user": "walking around Pike Place market and museums", "assistant": "listen"},
    {"user": "no voice", "assistant": "Pack a breathable waterproof rain shell"},
    {"user": "no voice", "assistant": "and sturdy comfortable water-resistant walking shoes."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00197"] = [
    {"user": "Can you give me a recipe", "assistant": "listen"},
    {"user": "for homemade creamy garlic hummus", "assistant": "listen"},
    {"user": "no voice", "assistant": "Blend cooked chickpeas with tahini,"},
    {"user": "no voice", "assistant": "lemon juice, garlic, and cumin."},
    {"user": "wait, can you help with interview prep", "assistant": "listen"},
    {"user": "answering what is your greatest weakness", "assistant": "listen"},
    {"user": "no voice", "assistant": "Certainly. What real area needs growth?"},
    {"user": "saying yes to too many projects", "assistant": "listen"},
    {"user": "no voice", "assistant": "Frame it as learning boundary prioritization"},
    {"user": "no voice", "assistant": "and delegating tasks systematically."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00386"] = [
    {"user": "How do airplanes generate lift", "assistant": "listen"},
    {"user": "over cambered aircraft wings", "assistant": "listen"},
    {"user": "no voice", "assistant": "Wing camber deflects airflow downward,"},
    {"user": "no voice", "assistant": "generating upward reactive pressure forces."},
    {"user": "actually, help me with resume bullets", "assistant": "listen"},
    {"user": "for a customer success manager role", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure. What key metrics did you achieve?"},
    {"user": "reduced enterprise client churn twenty percent", "assistant": "listen"},
    {"user": "no voice", "assistant": "Lead with that twenty percent retention gain"},
    {"user": "no voice", "assistant": "and detail the onboarding framework used."},
    {"user": "no voice", "assistant": "listen"},
]

# 20. user_interrupts_answer (8 items):
# Template length: 10 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:no voice | A:SPEAK
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:SPEAK | A:listen
# 7: U:no voice | A:SPEAK
# 8: U:no voice | A:SPEAK
# 9: U:no voice | A:listen

CONVERSATIONS["conv_00599"] = [
    {"user": "What is cloud computing", "assistant": "listen"},
    {"user": "no voice", "assistant": "Cloud computing delivers computing services"},
    {"user": "no voice", "assistant": "like servers storage and databases"},
    {"user": "hold on, what is serverless then", "assistant": "listen"},
    {"user": "no voice", "assistant": "Serverless runs event-driven code fragments"},
    {"user": "no voice", "assistant": "without provisioning virtual server instances."},
    {"user": "okay, go back to cloud basics", "assistant": "listen"},
    {"user": "no voice", "assistant": "Cloud providers maintain remote hardware,"},
    {"user": "no voice", "assistant": "charging only for actual capacity utilized."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00053"] = [
    {"user": "How does fermentation make cheese", "assistant": "listen"},
    {"user": "no voice", "assistant": "Lactic bacteria consume milk lactose,"},
    {"user": "no voice", "assistant": "converting sugars into lactic acid"},
    {"user": "wait, what role does rennet play", "assistant": "listen"},
    {"user": "no voice", "assistant": "Rennet contains enzymes that curdle proteins"},
    {"user": "no voice", "assistant": "separating solid curd from liquid whey."},
    {"user": "got it, please keep going", "assistant": "listen"},
    {"user": "no voice", "assistant": "Curds are salted pressed and aged,"},
    {"user": "no voice", "assistant": "developing distinct artisanal textures and flavors."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00032"] = [
    {"user": "Explain how GPS satellites work", "assistant": "listen"},
    {"user": "no voice", "assistant": "A constellation of orbital atomic satellites"},
    {"user": "no voice", "assistant": "continuously transmits precise timestamped radio signals"},
    {"user": "wait, what is trilateration exactly", "assistant": "listen"},
    {"user": "no voice", "assistant": "Trilateration calculates your geometric position"},
    {"user": "no voice", "assistant": "by intersecting signal sphere distances."},
    {"user": "understood, resume explaining GPS", "assistant": "listen"},
    {"user": "no voice", "assistant": "Your phone requires four satellite locks"},
    {"user": "no voice", "assistant": "to pinpoint exact altitude latitude longitude."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00578"] = [
    {"user": "What is an optical fiber", "assistant": "listen"},
    {"user": "no voice", "assistant": "Optical fibers are thin glass strands"},
    {"user": "no voice", "assistant": "carrying high-speed data as light pulses"},
    {"user": "hold on, what is total internal reflection", "assistant": "listen"},
    {"user": "no voice", "assistant": "Light bounces inside the glass core"},
    {"user": "no voice", "assistant": "because cladding reflects rays back entirely."},
    {"user": "okay, continue about fiber cables", "assistant": "listen"},
    {"user": "no voice", "assistant": "Fibers transmit vast terabytes across oceans"},
    {"user": "no voice", "assistant": "with virtually immune electromagnetic noise resistance."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00494"] = [
    {"user": "How do vaccines build immunity", "assistant": "listen"},
    {"user": "no voice", "assistant": "Vaccines introduce harmless viral antigens"},
    {"user": "no voice", "assistant": "training immune cells to recognize pathogens"},
    {"user": "wait, what are memory B-cells", "assistant": "listen"},
    {"user": "no voice", "assistant": "Memory B-cells retain pathogen blueprints"},
    {"user": "no voice", "assistant": "producing rapid antibodies upon future infection."},
    {"user": "I see, continue explaining immunity", "assistant": "listen"},
    {"user": "no voice", "assistant": "When genuine viruses invade later on,"},
    {"user": "no voice", "assistant": "your immune system clears them immediately."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00116"] = [
    {"user": "What is quantum computing", "assistant": "listen"},
    {"user": "no voice", "assistant": "Quantum computers leverage subatomic quantum physics"},
    {"user": "no voice", "assistant": "to process information exponentially faster"},
    {"user": "hold up, what is superposition", "assistant": "listen"},
    {"user": "no voice", "assistant": "Superposition allows qubits to exist simultaneously"},
    {"user": "no voice", "assistant": "in both zero and one states."},
    {"user": "got that, please proceed", "assistant": "listen"},
    {"user": "no voice", "assistant": "Entanglement links paired qubits together,"},
    {"user": "no voice", "assistant": "enabling complex optimization problem solving."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00536"] = [
    {"user": "How do trees absorb water", "assistant": "listen"},
    {"user": "no voice", "assistant": "Roots absorb subterranean moisture osmosis"},
    {"user": "no voice", "assistant": "transporting water upward via xylem vessels"},
    {"user": "wait, what drives water upward", "assistant": "listen"},
    {"user": "no voice", "assistant": "Transpiration pull evaporates leaf water"},
    {"user": "no voice", "assistant": "creating negative pressure through xylem columns."},
    {"user": "fascinating, continue the explanation", "assistant": "listen"},
    {"user": "no voice", "assistant": "Cohesion between hydrogen bonded water molecules"},
    {"user": "no voice", "assistant": "pulls unbroken streams hundreds of feet."},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00221"] = [
    {"user": "What is anaerobic exercise", "assistant": "listen"},
    {"user": "no voice", "assistant": "Anaerobic exercise involves high-intensity exertion"},
    {"user": "no voice", "assistant": "breaking down glucose without oxygen gas"},
    {"user": "hold on, what is lactic acid", "assistant": "listen"},
    {"user": "no voice", "assistant": "Lactate is an energy byproduct produced"},
    {"user": "no voice", "assistant": "when muscle glycogen breaks down anaerobically."},
    {"user": "alright, please continue", "assistant": "listen"},
    {"user": "no voice", "assistant": "Short intense sprints build fast-twitch muscle,"},
    {"user": "no voice", "assistant": "boosting power speed and explosive strength."},
    {"user": "no voice", "assistant": "listen"},
]

# 21. user_requests_pause (8 items):
# Template length: 9 turns
# Pattern:
# 0: U:SPEAK | A:listen
# 1: U:no voice | A:SPEAK
# 2: U:no voice | A:SPEAK
# 3: U:SPEAK | A:listen
# 4: U:no voice | A:SPEAK
# 5: U:no voice | A:SPEAK
# 6: U:no voice | A:SPEAK
# 7: U:SPEAK | A:listen
# 8: U:no voice | A:listen

CONVERSATIONS["conv_00503"] = [
    {"user": "Explain how encryption ciphers work", "assistant": "listen"},
    {"user": "no voice", "assistant": "Ciphers transform readable plaintext data"},
    {"user": "no voice", "assistant": "into scrambled unreadable ciphertext strings"},
    {"user": "pause a moment, let me write this down", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take all the time needed."},
    {"user": "no voice", "assistant": "Cryptographic keys unlock corresponding encrypted messages"},
    {"user": "no voice", "assistant": "using reversible mathematical modular transformations."},
    {"user": "okay, got that noted down", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00272"] = [
    {"user": "Explain the concept of inflation", "assistant": "listen"},
    {"user": "no voice", "assistant": "Inflation measures the broad sustained rise"},
    {"user": "no voice", "assistant": "in average prices across an economy"},
    {"user": "wait a second, let me digest that", "assistant": "listen"},
    {"user": "no voice", "assistant": "No rush at all."},
    {"user": "no voice", "assistant": "When money supply outpaces goods production,"},
    {"user": "no voice", "assistant": "each individual currency dollar buys less."},
    {"user": "okay, ready for more", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00377"] = [
    {"user": "How does compound interest accumulate", "assistant": "listen"},
    {"user": "no voice", "assistant": "Compound interest earns additional yield"},
    {"user": "no voice", "assistant": "on both principal and past interest"},
    {"user": "hold on, grab me a notebook first", "assistant": "listen"},
    {"user": "no voice", "assistant": "Sure, let me know when ready."},
    {"user": "no voice", "assistant": "Reinvesting quarterly returns accelerates portfolio growth"},
    {"user": "no voice", "assistant": "exponentially over multi-decade investment horizons."},
    {"user": "ready, I wrote it down", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00566"] = [
    {"user": "Explain how neural networks learn", "assistant": "listen"},
    {"user": "no voice", "assistant": "Networks adjust internal connection weights"},
    {"user": "no voice", "assistant": "to minimize prediction output error"},
    {"user": "wait, let me silence my phone", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take your time."},
    {"user": "no voice", "assistant": "Backpropagation computes mathematical gradients backwards,"},
    {"user": "no voice", "assistant": "updating millions of layer matrix parameters."},
    {"user": "alright, continue please", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00314"] = [
    {"user": "Explain how jet engines work", "assistant": "listen"},
    {"user": "no voice", "assistant": "Jet engines compress intake atmospheric air"},
    {"user": "no voice", "assistant": "mixing it with kerosene aviation fuel"},
    {"user": "hang on, someone is knocking at my door", "assistant": "listen"},
    {"user": "no voice", "assistant": "Go ahead, I will wait."},
    {"user": "no voice", "assistant": "Ignited gas expands through rear turbine,"},
    {"user": "no voice", "assistant": "producing massive forward propelling aerodynamic thrust."},
    {"user": "back now, all set", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00398"] = [
    {"user": "Explain what causes economic recessions", "assistant": "listen"},
    {"user": "no voice", "assistant": "Recessions occur when economic activity contracts"},
    {"user": "no voice", "assistant": "across consumer spending and corporate investment"},
    {"user": "hold on, I need to grab my glasses", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take your time finding them."},
    {"user": "no voice", "assistant": "Credit tightening and falling aggregate demand"},
    {"user": "no voice", "assistant": "reduce employment and overall GDP output."},
    {"user": "glasses on, I am following", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00482"] = [
    {"user": "Explain how kidney dialysis works", "assistant": "listen"},
    {"user": "no voice", "assistant": "Dialysis filters metabolic waste toxins"},
    {"user": "no voice", "assistant": "and excess fluid from bloodstream"},
    {"user": "wait, let me grab a glass of water", "assistant": "listen"},
    {"user": "no voice", "assistant": "Take all the time you need."},
    {"user": "no voice", "assistant": "Blood flows through semipermeable artificial membranes,"},
    {"user": "no voice", "assistant": "exchanging waste with specialized dialysate solution."},
    {"user": "all right, back and listening", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

CONVERSATIONS["conv_00041"] = [
    {"user": "How does gene editing with CRISPR work", "assistant": "listen"},
    {"user": "no voice", "assistant": "CRISPR uses guide RNA sequences"},
    {"user": "no voice", "assistant": "to target precise chromosomal DNA locations"},
    {"user": "hold up, let me open my biology notes", "assistant": "listen"},
    {"user": "no voice", "assistant": "Of course, take a moment."},
    {"user": "no voice", "assistant": "The Cas enzyme makes double-strand cuts,"},
    {"user": "no voice", "assistant": "allowing natural cellular repair or insertion."},
    {"user": "got notes open, continue please", "assistant": "listen"},
    {"user": "no voice", "assistant": "listen"},
]

print(f"Total conversations defined: {len(CONVERSATIONS)}")
