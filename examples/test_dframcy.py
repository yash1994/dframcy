from dframcy import DframCy

dfmc = DframCy("en_core_web_sm")
doc = dfmc.nlp(u"i am here in USA")
print(dfmc.to_dataframe(doc))
