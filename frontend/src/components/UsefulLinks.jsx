const links = [
  { name: 'Github Repo', url: 'https://github.com/Jamandella/magic_stats' },
  { name: 'Google Style Guide', url: 'https://developers.google.com/style' },
  { name: 'Figma Wireframe', url: 'https://www.figma.com/file/YqYaigEWARHy5NL5g08Kpq/Magic-Stats-Project?type=whiteboard&t=UKJd6ZK53VYmdl8B-0' },
  { name: 'Meeting Notes', url: 'https://drive.google.com/drive/folders/126ZhLhKfrLapQaeJPCU4REnxilpgR3qA' },
  { name: 'Lucid Chart', url: 'https://lucid.app/lucidchart/002d3ba4-b5e1-44b8-9c6c-3b6e62869340/edit?invitationId=inv_d3a938cc-a15e-49bf-8d0a-d7902a952bda&page=0_0#' },
  { name: '17 Lands', url: 'https://www.17lands.com/' },
  { name: 'Recharts', url: 'https://recharts.org/en-US/' },
  { name: 'Trello Workspace', url: 'https://trello.com/b/z2cY9T70/magic-project-tracker' },
];

const UsefulLinks = () => {
  return (
    <div className="flow">
      <h2 className="heading-3">Useful Links</h2>
      <ul>
        {links.map(link => (
          <li key={link.name}>
            <a className="link" href={link.url} target="_blank">{link.name}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default UsefulLinks;
