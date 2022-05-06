db.createUser(
    {
        user : 'root',
        pwd : 'passwd',
        roles : [{
                role : 'readWrite',
                db : 'Token_prices'
            }],
			});