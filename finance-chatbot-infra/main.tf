provider "azurerm" {
  features {}
  subscription_id = "subscription_id"
}

resource "azurerm_resource_group" "rg" {
  name     = "financio-rg"
  location = "Central US"
}

resource "azurerm_mssql_server" "sql_server" {
  depends_on = [azurerm_resource_group.rg]
  name                         = "financio-sql-server"
  resource_group_name          = azurerm_resource_group.rg.name
  location                     = azurerm_resource_group.rg.location
  version                      = "12.0"
  administrator_login          = "administrator_login"
  administrator_login_password = "administrator_login_password"
}


resource "azurerm_mssql_database" "db" {
  name             = "financio-db"
  server_id        = azurerm_mssql_server.sql_server.id
  collation        = "SQL_Latin1_General_CP1_CI_AS"
  license_type     = "LicenseIncluded"
  max_size_gb      = 2
  sku_name         = "Basic"
}

output "database_connection_string" {
  value     = "Server=tcp:${azurerm_mssql_server.sql_server.fully_qualified_domain_name},1433;Initial Catalog=${azurerm_mssql_database.db.name};Persist Security Info=False;User ID=financioadmin;Password=YourStrongPassword123!;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
  sensitive = true
}
