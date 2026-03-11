# real time application of lists
# to do list app

to_do_list = ["cook breakfast", "buy milk", "walking", "homework", "tv watching", "pay bills"]
print("Initial todo list items: ", to_do_list)

#To add a task
to_do_list.append("sleep")
to_do_list.append("call a friend")

#to remove completed task
to_do_list.remove("buy milk")

#check if atask is in list and send reminder
if "pay bills" in to_do_list:
        print("Don't forget to pay bills")

print("To do list remaining:")
for task in to_do_list:
    print(f"-{task}")